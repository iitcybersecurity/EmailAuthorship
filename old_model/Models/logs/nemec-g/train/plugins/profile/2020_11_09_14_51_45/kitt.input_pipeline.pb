	�a0��J@�a0��J@!�a0��J@	~�s���?~�s���?!~�s���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�a0��J@N�g\W�?A�> �M�J@Y���XP�?*	/�$�5^@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��`R|�?!�!_&P�@@)�K⬈��?1;�R�:@:Preprocessing2F
Iterator::ModelL8��+�?!~����C@)�O7P���?1&w� 7@:Preprocessing2U
Iterator::Model::ParallelMapV2��8�j��?!��@��/@)��8�j��?1��@��/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���2#�?!�38�P-@)�N�6���?1�iH��"@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap$��:��?!�A���v5@)*��D؀?1�� �:@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.��Hٲ?!�8�mQwN@)iUMu?1�7VBl@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorÜ�M?!6�6@)Ü�M?16�6@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	^�z?!-��mx@)	^�z?1-��mx@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9~�s���?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N�g\W�?N�g\W�?!N�g\W�?      ��!       "      ��!       *      ��!       2	�> �M�J@�> �M�J@!�> �M�J@:      ��!       B      ��!       J	���XP�?���XP�?!���XP�?R      ��!       Z	���XP�?���XP�?!���XP�?JCPU_ONLYY~�s���?b 