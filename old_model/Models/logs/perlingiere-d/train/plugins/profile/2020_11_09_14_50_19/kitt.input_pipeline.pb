	�C�HJN@�C�HJN@!�C�HJN@	��#3��?��#3��?!��#3��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�C�HJN@��c${��?AL�u�4N@Y��we�?*	j�t�(\@2F
Iterator::Modelfj�!��?!v���DG@)ʈ@�t�?1�L�d�D>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���0�?!�8QD�=@)9ӄ�'c�?1���Hʜ8@:Preprocessing2U
Iterator::Model::ParallelMapV27���0�?!���n�/@)7���0�?1���n�/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�����M�?!9B�rDF*@)��'�ځ?11&�q��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���2W�?!�uX�^3@)�̯� �|?1"��{(�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?#K�x?!A^Ut��@)?#K�x?1A^Ut��@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�m�(�?!�mF��J@)�����w?1�@@��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��J?��v?!eu��s�@)��J?��v?1eu��s�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9��#3��?#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��c${��?��c${��?!��c${��?      ��!       "      ��!       *      ��!       2	L�u�4N@L�u�4N@!L�u�4N@:      ��!       B      ��!       J	��we�?��we�?!��we�?R      ��!       Z	��we�?��we�?!��we�?JCPU_ONLYY��#3��?b 