�	�D���0@�D���0@!�D���0@	m?Y�^�?m?Y�^�?!m?Y�^�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�D���0@���6�ٳ?A�[;Q�0@Y���n��?*	j�t��a@2F
Iterator::ModelEH�ξ�?!�-|J_�G@)稣�j�?1�ϝ}n<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat2��У?!e琵;�;@)ya�X5�?1��0��6@:Preprocessing2U
Iterator::Model::ParallelMapV2�R#�3��?!��Z��2@)�R#�3��?1��Z��2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate����=�?!kݲ�D2@)RE�*k��?1�Q���)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Ϲ���?![҃��fJ@)t(CUL�?1�[8@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS"�~?!�lҸaB@)S"�~?1�lҸaB@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�� Z+�|?! ���V@)�� Z+�|?1 ���V@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�a���L�?!N�����3@)#�ng_y`?1�Ҕ�^��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9m?Y�^�?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���6�ٳ?���6�ٳ?!���6�ٳ?      ��!       "      ��!       *      ��!       2	�[;Q�0@�[;Q�0@!�[;Q�0@:      ��!       B      ��!       J	���n��?���n��?!���n��?R      ��!       Z	���n��?���n��?!���n��?JCPU_ONLYYm?Y�^�?b Y      Y@q�.5;���?"�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 