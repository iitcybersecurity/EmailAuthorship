�	I����.@I����.@!I����.@	GJ�0���?GJ�0���?!GJ�0���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$I����.@�ZC����?A��Q��-@Y9����?*	9��v�/a@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�&p�n�?!^��E��E@)���N�?1���<D@:Preprocessing2F
Iterator::Model�H�F�q�?!�V#�Nv>@)N*kg�?1��aB�v3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��HK��?!Ek/ܴ1@)%=�Nΐ?1ڐ�k��'@:Preprocessing2U
Iterator::Model::ParallelMapV2���C���?!^f����%@)���C���?1^f����%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	4��y�?!H*7]lbQ@)e�I)���?1���g��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor-y<-?�?!`���C@)-y<-?�?1`���C@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��ٕf?!�D�
 @)��ٕf?1�D�
 @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?q ���?!U��5X�F@)�óe?1�~�^��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��?�V?!Z���>�?)��?�V?1Z���>�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9GJ�0���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�ZC����?�ZC����?!�ZC����?      ��!       "      ��!       *      ��!       2	��Q��-@��Q��-@!��Q��-@:      ��!       B      ��!       J	9����?9����?!9����?R      ��!       Z	9����?9����?!9����?JCPU_ONLYYGJ�0���?b Y      Y@q�c�}�H�?"�
device�Your program is NOT input-bound because only 0.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 