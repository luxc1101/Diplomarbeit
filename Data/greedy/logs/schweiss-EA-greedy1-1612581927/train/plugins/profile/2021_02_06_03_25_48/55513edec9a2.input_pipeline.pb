	�c@�z�@�c@�z�@!�c@�z�@	��nԴ^@��nԴ^@!��nԴ^@"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�c@�z�@�l���?1^f�(��?A�NGɫ�?Id����?Y;�G�?*	�MbXQ[@2U
Iterator::Model::ParallelMapV2ܡa1�Z�?!�>��=LA@)ܡa1�Z�?1�>��=LA@:Preprocessing2F
Iterator::Model�1 Ǟ�?!�b���xJ@)�E����?1rH��Y2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Z&��|�?! [pf4@)�k���P�?1.H�)-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��)Ւ?!�� N��0@)�4a���?1P���+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipq:�V�S�?!�]&8�G@)8�Jw�ـ?1�nў@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceZ���аx?!%0�	�@)Z���аx?1%0�	�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor]���Ej?!���s�z@)]���Ej?1���s�z@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!�:J$�6@)�1��Ag?16�R��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 28.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�52.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��nԴ^@IG�E-�T@Q��<w+@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�l���?�l���?!�l���?      ��!       "	^f�(��?^f�(��?!^f�(��?*      ��!       2	�NGɫ�?�NGɫ�?!�NGɫ�?:	d����?d����?!d����?B      ��!       J	;�G�?;�G�?!;�G�?R      ��!       Z	;�G�?;�G�?!;�G�?b      ��!       JGPUY��nԴ^@b qG�E-�T@y��<w+@