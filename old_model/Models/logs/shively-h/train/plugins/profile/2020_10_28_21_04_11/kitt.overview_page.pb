�	�,_��,@�,_��,@!�,_��,@	5�V�_��?5�V�_��?!5�V�_��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�,_��,@��=^H��?A'ݖ�G,@Y�$���?*	�S㥛6|@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapۤ����?!M;��wF@) �yrM��?1��Jl�A@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map겘�|\�?!����|B@)����`��?1_=%���<@:Preprocessing2F
Iterator::ModelZd;�O�?!�6wc])@)��`�H�?1�:J@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat��z�p̢?!�c�cD @)�6T��7�?1�+5��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateL�k�˘?!�hk�t@)�
(�ӗ?1���~j�@:Preprocessing2U
Iterator::Model::ParallelMapV2T^-w�?!�N1�p@)T^-w�?1�N1�p@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�*8� �?!r_Dܜ_@)��:8؛�?1�\6VK@:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch���_w��?!���\@)���_w��?1���\@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip=֌r�?!P�~�,I@)�W\�{?1p!�@��?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�2p@Kw?!�;(�?)�2p@Kw?1�;(�?:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�b)���?!�P�z/�?)�>��Vv?1u���T�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice�iT�dk?!�Gׂ��?)�iT�dk?1�Gׂ��?:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range�"j��Gi?!��|N��?)�"j��Gi?1��|N��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��<�N?!Y؋���?)��<�N?1Y؋���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no95�V�_��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��=^H��?��=^H��?!��=^H��?      ��!       "      ��!       *      ��!       2	'ݖ�G,@'ݖ�G,@!'ݖ�G,@:      ��!       B      ��!       J	�$���?�$���?!�$���?R      ��!       Z	�$���?�$���?!�$���?JCPU_ONLYY5�V�_��?b Y      Y@q�I2�{��?"�
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