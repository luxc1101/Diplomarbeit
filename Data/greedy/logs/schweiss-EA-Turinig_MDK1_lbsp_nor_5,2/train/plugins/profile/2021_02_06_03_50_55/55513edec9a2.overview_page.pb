�	Na����\@Na����\@!Na����\@	�\-�v;�?�\-�v;�?!�\-�v;�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Na����\@��$���[@1F'K���?A�J#f�y�?I�T�=� @Y��gyܱ?*	��|?5.b@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�DkE��?!�"�M7�B@)��2nj��?1���5�4A@:Preprocessing2U
Iterator::Model::ParallelMapV2��Aȗ�?!���5�@@)��Aȗ�?1���5�@@:Preprocessing2F
Iterator::ModelK#f�y��?!����H@)h��W�?1�+y��0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip\ qW��?!Zx��hI@)d��uy?1�&�a@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����ׁ�?!3�2@)�#c��u?1�8� �@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor>���q?!�	f�@)>���q?1�	f�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapٕ��zO�?!%-T��C@)�'eRCk?1��P�)N@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor��cw�b?!d�#���?)��cw�b?1d�#���?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�]�pXZ?!�&��t��?)�]�pXZ?1�&��t��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 97.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9�\-�v;�?I��=���X@Q�����+�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��$���[@��$���[@!��$���[@      ��!       "	F'K���?F'K���?!F'K���?*      ��!       2	�J#f�y�?�J#f�y�?!�J#f�y�?:	�T�=� @�T�=� @!�T�=� @B      ��!       J	��gyܱ?��gyܱ?!��gyܱ?R      ��!       Z	��gyܱ?��gyܱ?!��gyܱ?b      ��!       JGPUY�\-�v;�?b q��=���X@y�����+�?�"L
.gradient_tape/fein_turning/enc1/dense_2/MatMulMatMul��*��6�?!��*��6�?0"E
)gradient_tape/fein_turning/dense/MatMul_1MatMul�U�+%��?!R�x�?"L
0gradient_tape/fein_turning/enc2/dense_4/MatMul_1MatMul\��T{�?! �4B��?"R
/gradient_tape/binary_crossentropy/DynamicStitchDynamicStitch��2����?!�BAn���?">
 fein_turning/enc2/dense_4/MatMulMatMulK�,`B�?!<r�o�5�?0">
 fein_turning/enc1/dense_2/MatMulMatMulT�:�LԔ?!���T5�?0"E
'gradient_tape/fein_turning/dense/MatMulMatMulT�:�LԔ?!���m���?0"L
.gradient_tape/fein_turning/enc2/dense_4/MatMulMatMulT�:�LԔ?!|O�hj�?0"7
fein_turning/dense/MatMulMatMul~JR�V�?!̘F����?0"W
6gradient_tape/fein_turning/dense_1/BiasAdd/BiasAddGradBiasAddGrad�ܳ���?!l��n��?Q      Y@Y�B���0@aNozӛ�T@q�iA��W@y�I`	�?"�

both�Your program is POTENTIALLY input-bound because 97.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 