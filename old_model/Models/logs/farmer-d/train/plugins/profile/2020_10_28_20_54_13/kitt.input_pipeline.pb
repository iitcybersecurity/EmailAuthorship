	>�4a�6@>�4a�6@!>�4a�6@	��b����?��b����?!��b����?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$>�4a�6@�u��$�?A��-�5@Y�2���?*	=
ףpA`@2F
Iterator::ModelJ+��?!l��&��B@)h^���?1�����6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM��f�ס?!��z�2�:@)~b��U�?1�V�6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateG�J���?!�5��~:@)�X�vM�?1���J�?2@:Preprocessing2U
Iterator::Model::ParallelMapV2,����?!O޶��'.@),����?1O޶��'.@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,����?!�wl-| @),����?1�wl-| @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�x]��?!�TG�)O@)+�&�|��?1���6g@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorrl=C8fy?!�i�0�@)rl=C8fy?1�i�0�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapc��K�A�?!F\��<@)��� �i?1F%~p@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��b����?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�u��$�?�u��$�?!�u��$�?      ��!       "      ��!       *      ��!       2	��-�5@��-�5@!��-�5@:      ��!       B      ��!       J	�2���?�2���?!�2���?R      ��!       Z	�2���?�2���?!�2���?JCPU_ONLYY��b����?b 