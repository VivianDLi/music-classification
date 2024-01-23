package org.tensorflow.lite.examples.audio;

import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.ConstantQ;
import timber.log.Timber;

public class InputProcessor implements AudioProcessor {
    protected final float sampleRate;
    protected final int bufferSize;
    protected final ConstantQ cqt;
    protected ByteBuffer outputBuffer;
    protected ByteBuffer foregroundBuffer;
    protected ByteBuffer backgroundBuffer;

    public InputProcessor(float sampleRate, int bufferSize, float inferenceSeconds, ByteBuffer outputBuffer, ConstantQ cqt) {
        this.sampleRate = sampleRate;
        this.bufferSize = bufferSize;
        this.cqt = cqt;
        this.outputBuffer = outputBuffer;
        int outputBufferSize = (int) Math.ceil(84 * 2 * (sampleRate * inferenceSeconds / bufferSize)) * 4;
        this.foregroundBuffer = ByteBuffer.allocateDirect(outputBufferSize).order(ByteOrder.nativeOrder());
        this.backgroundBuffer = ByteBuffer.allocateDirect(outputBufferSize).order(ByteOrder.nativeOrder());
    }
    @Override
    public boolean process(AudioEvent audioEvent) {
        float[] magnitudeBuffer = cqt.getMagnitudes();
        System.out.println(magnitudeBuffer.length);
        FloatBuffer tempBuffer = this.backgroundBuffer.asFloatBuffer();
        tempBuffer.put(magnitudeBuffer);
        if (this.backgroundBuffer.limit() == this.backgroundBuffer.capacity()) {
            try {
                // swap background and foreground
                this.foregroundBuffer.clear();
                this.foregroundBuffer.put(this.backgroundBuffer);
                this.backgroundBuffer.clear();
                this.outputBuffer = this.transpose(this.foregroundBuffer);
            } catch (BufferOverflowException e) {
                Timber.tag("InputProcessor").e("Buffer overflow exception with error %s, resetting all buffers.", e.getMessage());
                e.printStackTrace();
                this.foregroundBuffer.clear();
                this.backgroundBuffer.clear();
            } catch (UnsupportedOperationException e) {
                Timber.tag("InputProcessor").e("Failed to transpose foreground buffer with error %s.", e.getMessage());
                e.printStackTrace();
            }
        }
        return true;
    }

    @Override
    public void processingFinished() {
    }

    private ByteBuffer transpose(ByteBuffer array) {
        int m = array.capacity() / (4 * 84);
        int n = 84;
        ByteBuffer ret = ByteBuffer.allocate(array.capacity()).order(ByteOrder.nativeOrder());
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                ret.putFloat((i * m + j) * 4, array.getFloat((j * n + i) * 4));
            }
        }
        return ret;
    }
}
