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

package org.tensorflow.lite.examples.audio.fragments

import android.annotation.SuppressLint
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import org.tensorflow.lite.examples.audio.AudioClassificationHelper
import org.tensorflow.lite.examples.audio.R
import org.tensorflow.lite.examples.audio.databinding.FragmentAudioBinding
import org.tensorflow.lite.examples.audio.ui.DistancesAdapter

interface AudioClassificationListener {
    fun onError(error: String)
    fun onResult(results: List<MutableMap.MutableEntry<Float, String>>, inferenceTime: Long, classifyTime: Long)
}

class AudioFragment : Fragment() {
    private var _fragmentBinding: FragmentAudioBinding? = null
    private val fragmentAudioBinding get() = _fragmentBinding!!
    private val adapter by lazy { DistancesAdapter() }

    private lateinit var audioHelper: AudioClassificationHelper

    private var modelName: String = "full_model"
    private var quantized: Boolean = false
    private var pruned: Boolean = false

    private val audioClassificationListener = object : AudioClassificationListener {
        @SuppressLint("NotifyDataSetChanged")
        override fun onResult(results: List<MutableMap.MutableEntry<Float, String>>, inferenceTime: Long, classifyTime: Long) {
            requireActivity().runOnUiThread {
                adapter.referencesList = results
                adapter.notifyDataSetChanged()
                fragmentAudioBinding.bottomSheetLayout.inferenceTimeVal.text =
                    String.format("%d ms", inferenceTime)
                fragmentAudioBinding.bottomSheetLayout.classifyTimeVal.text = String.format("%d ms", classifyTime)
            }
        }

        @SuppressLint("NotifyDataSetChanged")
        override fun onError(error: String) {
            requireActivity().runOnUiThread {
                Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
                adapter.referencesList = emptyList()
                adapter.notifyDataSetChanged()
            }
        }
    }

    override fun onCreateView(
      inflater: LayoutInflater,
      container: ViewGroup?,
      savedInstanceState: Bundle?
    ): View {
        _fragmentBinding = FragmentAudioBinding.inflate(inflater, container, false)
        return fragmentAudioBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        fragmentAudioBinding.recyclerView.adapter = adapter
        audioHelper = AudioClassificationHelper(
            requireContext(),
            audioClassificationListener
        )

        val spinner: Spinner = view.findViewById(R.id.reference_file)
        // Create an ArrayAdapter using the string array and a default spinner layout.
        ArrayAdapter.createFromResource(
            this.requireContext(),
            R.array.reference_files,
            android.R.layout.simple_spinner_item
        ).also { adapter ->
            // Specify the layout to use when the list of choices appears.
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            // Apply the adapter to the spinner.
            spinner.adapter = adapter
        }

        // Allow the user to select between multiple supported audio models.
        // The original location and documentation for these models is listed in
        // the `download_model.gradle` file within this sample. You can also create your own
        // audio model by following the documentation here:
        // https://www.tensorflow.org/lite/models/modify/model_maker/speech_recognition
        fragmentAudioBinding.bottomSheetLayout.modelSelector.setOnCheckedChangeListener { _, checkedId ->
            when (checkedId) {
                R.id.full_model -> {
                    audioHelper.stopAudioClassification()
                    modelName = "full_model"
                    audioHelper.currentModel =
                        (if (quantized) "quantized_" else "") + (if (pruned) "pruned_" else "") + modelName
                    audioHelper.initClassifier()
                }

                R.id.small_model -> {
                    audioHelper.stopAudioClassification()
                    modelName = "small_model"
                    audioHelper.currentModel =
                        (if (quantized) "quantized_" else "") + (if (pruned) "pruned_" else "") + modelName
                    audioHelper.initClassifier()
                }
            }
        }
        fragmentAudioBinding.bottomSheetLayout.quantizedToggle.setOnCheckedChangeListener { _, isChecked ->
            audioHelper.stopAudioClassification()
            quantized = isChecked
            audioHelper.currentModel =
                (if (quantized) "quantized_" else "") + (if (pruned) "pruned_" else "") + modelName
            audioHelper.initClassifier()
        }
        fragmentAudioBinding.bottomSheetLayout.prunedToggle.setOnCheckedChangeListener { _, isChecked ->
            audioHelper.stopAudioClassification()
            pruned = isChecked
            audioHelper.currentModel =
                (if (quantized) "quantized_" else "") + (if (pruned) "pruned_" else "") + modelName
            audioHelper.initClassifier()
        }
        fragmentAudioBinding.bottomSheetLayout.referenceFile.onItemSelectedListener = object: AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
                audioHelper.stopAudioClassification()
                audioHelper.referenceFile = parent.getItemAtPosition(position).toString()
                audioHelper.initClassifier()
            }
            override fun onNothingSelected(parent: AdapterView<*>) {

            }
        }
        // Allow the user to change the max number of results returned by the audio classifier.
        // Currently allows between 1 and 5 results, but can be edited here.
        fragmentAudioBinding.bottomSheetLayout.resultsMinus.setOnClickListener {
            if (audioHelper.numOfResults > 1) {
                audioHelper.numOfResults--
                audioHelper.stopAudioClassification()
                audioHelper.initClassifier()
                fragmentAudioBinding.bottomSheetLayout.resultsValue.text =
                    audioHelper.numOfResults.toString()
            }
        }
        fragmentAudioBinding.bottomSheetLayout.resultsPlus.setOnClickListener {
            if (audioHelper.numOfResults < 16) {
                audioHelper.numOfResults++
                audioHelper.stopAudioClassification()
                audioHelper.initClassifier()
                fragmentAudioBinding.bottomSheetLayout.resultsValue.text =
                    audioHelper.numOfResults.toString()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        // Make sure that all permissions are still present, since the
        // user could have removed them while the app was in paused state.
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(AudioFragmentDirections.actionAudioToPermissions())
        }

        if (::audioHelper.isInitialized ) {
            audioHelper.startAudioClassification()
        }
    }

    override fun onPause() {
        super.onPause()
        if (::audioHelper.isInitialized ) {
            audioHelper.stopAudioClassification()
        }
    }

    override fun onDestroyView() {
        _fragmentBinding = null
        super.onDestroyView()
    }
}
