�  *��~j��@)      �=2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapq<��*9@!��i�H@)n�y9@1�o��H@:Preprocessing2T
Iterator::Root::ParallelMapV2C�O�}~8@!,kgC*H@)C�O�}~8@1,kgC*H@:Preprocessing2t
=Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map:d�w�?!1���'�?)|�wJ�?1c���M��?:Preprocessing2�
KIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat�k�,	P�?!�|�-��?)G�˵h�?13�d~ñ?:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate�����?!�&q5ԛ�?)-��a�?1w���'��?:Preprocessing2E
Iterator::Root��w�8@!8��2-H@)6\�-ˇ?1�`X�[y�?:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�a��c�?!�~����?)�6�^��?1���}8H�?:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate�@�v�?!K{U&�!�?)"O���|�?1A�4q�9�?:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�]h��09@!��-4�H@)��%VF#?1��8�3��?:Preprocessing2o
8Iterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch����/�~?!݄��?)����/�~?1݄��?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���)o?!\ů�h�~?)���)o?1\ů�h�~?:Preprocessing2�
RIterator::Root::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range��6p�d?!RKn��t?)��6p�d?1RKn��t?:Preprocessing2�
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor?�{�&Z?!}���0�i?)?�{�&Z?1}���0�i?:Preprocessing2�
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor���k�6L?!R!�.��[?)���k�6L?1R!�.��[?:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice����KK?!�C�R��Z?)����KK?1�C�R��Z?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q�C���?"�
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B��No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.