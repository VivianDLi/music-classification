/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *             http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.audio.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ProgressBar
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.audio.R

class DistancesAdapter() : RecyclerView.Adapter<DistancesAdapter.ViewHolder>() {
    var referencesList: List<MutableMap.MutableEntry<Float, String>> = emptyList()

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val textView: TextView
        val progressBar: ProgressBar

        init {
            // Define click listener for the ViewHolder's View
            textView = view.findViewById(R.id.label_text_view)
            progressBar = view.findViewById(R.id.progress_bar)
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_distance, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val distanceEntry = referencesList[position]
        holder.textView.text = distanceEntry.value
        holder.progressBar.progress = (distanceEntry.key * 100).toInt()
    }

    override fun getItemCount(): Int {
        return referencesList.size
    }
}
