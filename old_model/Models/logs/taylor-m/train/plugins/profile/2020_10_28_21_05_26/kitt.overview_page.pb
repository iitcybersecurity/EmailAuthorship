�	+�C3O�7@+�C3O�7@!+�C3O�7@	uj09D�?uj09D�?!uj09D�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+�C3O�7@C �8�@�?A��c���7@Y�Χ�U�?*	��Q��]@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat6sHj�d�?!e��b>@)����~�?1����/9@:Preprocessing2F
Iterator::Model����G��?!����A@)�Ye����?1��.U�2@:Preprocessing2U
Iterator::Model::ParallelMapV2I��r�S�?!�����F.@)I��r�S�?1�����F.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateY32�]��?!./��5@)�!p$А?1�5�2Q�+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaphwH1@��?!�pv�h!=@)�v�4E��?1��Њ @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�� �rh�?!�P���@)�� �rh�?1�P���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�$y��?!����wP@)̛õ��~?1��&Wi@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�P�,y?!�ѯ8�@)�P�,y?1�ѯ8�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9uj09D�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	C �8�@�?C �8�@�?!C �8�@�?      ��!       "      ��!       *      ��!       2	��c���7@��c���7@!��c���7@:      ��!       B      ��!       J	�Χ�U�?�Χ�U�?!�Χ�U�?R      ��!       Z	�Χ�U�?�Χ�U�?!�Χ�U�?JCPU_ONLYYuj09D�?b Y      Y@q��VJ��?"�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 