/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.audio

import android.content.Context
import android.content.res.AssetFileDescriptor
import android.os.SystemClock
import be.tarsos.dsp.AudioDispatcher
import be.tarsos.dsp.AudioProcessor
import be.tarsos.dsp.ConstantQ
import be.tarsos.dsp.io.android.AudioDispatcherFactory
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.gpu.support.TfLiteGpu
import com.google.android.gms.tflite.java.TfLite
import com.google.gson.Gson
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.examples.audio.fragments.AudioClassificationListener
import org.tensorflow.lite.gpu.GpuDelegateFactory
import timber.log.Timber
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel
import java.util.TreeMap
import java.util.concurrent.ScheduledThreadPoolExecutor
import java.util.concurrent.TimeUnit
import kotlin.math.ceil
import kotlin.math.pow


class AudioClassificationHelper(
    private val context: Context,
    private val listener: AudioClassificationListener,
    var currentModel: String = MODEL_NAME,
    var referenceFile: String = REFERENCE_NAME,
    var quantizedModel: Boolean = QUANTIZED,
    private val sampleRate: Int = SAMPLE_RATE,
    private val hopLength: Int = HOP_LENGTH,
    private val inferenceSeconds: Float = INFERENCE_SECONDS,
    var numOfResults: Int = DEFAULT_NUM_OF_RESULTS,
) {
    private lateinit var interpreter: Interpreter
    private lateinit var dispatcher: AudioDispatcher
    private lateinit var cqtProcessor: ConstantQ
    private lateinit var inputProcessor: InputProcessor
    private lateinit var dispatcherThread: Thread
    private lateinit var modelExecutor: ScheduledThreadPoolExecutor


    private lateinit var modelOutput: ByteBuffer

    private var cqtBuffer: ByteBuffer = ByteBuffer.allocate(
        ceil(84 * (sampleRate * inferenceSeconds / hopLength)).toInt() * 4)
    private var modelResults: TreeMap<Float, String> = TreeMap<Float, String>()
    // Loading references to compare against
    val gson = Gson()
    val json = context.assets.open(referenceFile).bufferedReader().use { it.readText() }
    private var modelReferences: HashMap<String, Array<Float>> = gson.fromJson(json, HashMap<String, Array<Float>>()::class.java)

    private val classifyRunnable = Runnable {
        classifyAudio()
    }

    init {
        initClassifier()
    }

    fun initClassifier() {
        // Initialize GPU
        val useGpuTask = TfLiteGpu.isGpuDelegateAvailable(context)
        val interpreterTask = useGpuTask.continueWith { useGPU ->
            TfLite.initialize(context,
                TfLiteInitializationOptions.builder()
                    .setEnableGpuDelegateSupport(useGPU.result)
                    .build())
        }

        // Initialize Model
        val options = Interpreter.Options()
            .setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
            .addDelegateFactory(GpuDelegateFactory())
        try {
            interpreter = Interpreter(loadModelFile(), options);
            startAudioClassification()
        } catch (e: Exception) {
            listener.onError("TFLite failed to load model. See error logs for more details.")
            Timber.tag("AudioClassification").e("TFLite failed to load with error: %s", e.message)
        }
        modelOutput = ByteBuffer.allocateDirect(300 * java.lang.Float.SIZE / java.lang.Byte.SIZE).order(ByteOrder.nativeOrder())

        // Initialize Delegate
        // minFreq and highFreq set to C1 (increased slightly to get 84 bins) and C8
        cqtProcessor = ConstantQ(sampleRate.toFloat(), 32.71f, 4186f, 12f)
        val fftLength = cqtProcessor.ffTlength
        dispatcher = AudioDispatcherFactory.fromDefaultMicrophone(
            sampleRate, fftLength, hopLength
        )
        inputProcessor = InputProcessor(sampleRate.toFloat(), hopLength, inferenceSeconds, cqtBuffer, cqtProcessor)
        dispatcher.addAudioProcessor(cqtProcessor)
        dispatcher.addAudioProcessor(inputProcessor)
        dispatcherThread = Thread(dispatcher, "Sound processor")
    }
    @Throws(IOException::class)
    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor: AssetFileDescriptor = context.assets.openFd("models/$MODEL_NAME")
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declareLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declareLength)
    }

    fun startAudioClassification() {
        if (dispatcherThread.isAlive) {
            return
        }
        dispatcher.run()
        modelExecutor = ScheduledThreadPoolExecutor(1)
        val interval = (inferenceSeconds * 1000).toLong()
        modelExecutor.scheduleAtFixedRate(
            classifyRunnable,
            0,
            interval,
            TimeUnit.MILLISECONDS
        )
    }

    private fun classifyAudio() {
        val inputReadonly: ByteBuffer = cqtBuffer.asReadOnlyBuffer()
        val input: ByteBuffer
        if (quantizedModel) {
            input = ByteBuffer.allocate(inputReadonly.capacity() / 4)
            val params: Tensor.QuantizationParams = interpreter.getInputTensor(0).quantizationParams()
            for (i in 0..inputReadonly.capacity()) {
                // quantize inputs (input / scale + zero_point) as int8
                val value = inputReadonly.getFloat() / params.scale + params.zeroPoint
                input.put(value.toInt().toByte())
            }
        } else {
            input = ByteBuffer.allocate(inputReadonly.capacity())
            input.put(inputReadonly)
        }
        val startTime = SystemClock.uptimeMillis()
        interpreter.run(input, modelOutput)
        val inferenceTime = SystemClock.uptimeMillis() - startTime
        val representation: FloatBuffer = modelOutput.asFloatBuffer()
        this.calculateDistances(representation)
        val classifyTime = SystemClock.uptimeMillis() - startTime
        val filteredResults = modelResults.entries.take(numOfResults)
        listener.onResult(filteredResults, inferenceTime, classifyTime)
    }

    private fun calculateDistances(query: FloatBuffer) {
        modelReferences.forEach { (song, reference) ->
            var dot: Float = 0f
            var rMag: Float = 0f
            var qMag: Float = 0f
            for (i in 0..query.capacity()) {
                dot += query.get(i) * reference.get(i)
                rMag += reference.get(i).pow(2f)
                qMag += query.get(i).pow(2f)
            }
            rMag = rMag.pow(0.5f)
            qMag = qMag.pow(0.5f)
            modelResults[dot / (rMag * qMag)] = song
        }
    }

    fun stopAudioClassification() {
        dispatcherThread.interrupt();
        dispatcher.stop()
        modelExecutor.shutdownNow();
        interpreter.close()
    }

    companion object {
        const val SAMPLE_RATE = 22050
        const val HOP_LENGTH = 512
        const val INFERENCE_SECONDS = 2f
        const val DEFAULT_NUM_OF_RESULTS = 10
        const val QUANTIZED = true
        const val MODEL_NAME = "full_model.tflite"
        const val REFERENCE_NAME = "full_model-mazurkas.json"
    }
}
