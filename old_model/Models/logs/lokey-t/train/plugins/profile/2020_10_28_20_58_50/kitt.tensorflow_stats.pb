"��
BHostIDLE"IDLE1H�z��@AH�z��@a�������?i�������?�Unknown
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1%��?�@9%��?�@A%��?�@I%��?�@a��,�w�?i ���C�?�Unknown
lHostConv2D"sequential/conv1d/conv1d(1��/�@9��/�@A��/�@I��/�@a*Sf���?i�U�#�{�?�Unknown
�HostConv2DBackpropInput":gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropInput(1��K�e�@9��K�e�@A��K�e�@I��K�e�@a!�gOP�?i�S����?�Unknown
�HostCast"2gradient_tape/sequential/global_max_pooling1d/Cast(1     �v@9     �v@A     �v@I     �v@a؜��$7�?iu�:6dW�?�Unknown
�HostEqual"3gradient_tape/sequential/global_max_pooling1d/Equal(1�Zd;q@9�Zd;q@A�Zd;q@I�Zd;q@a>��J"c�?i��f����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1��Q��m@9��Q��m@A��Q��m@I��Q��m@a 8̻��?i� V"�#�?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(19��v�m@99��v�m@A9��v�m@I9��v�m@a�y؅�J�?i��m�ˀ�?�Unknown
n	HostBiasAdd"sequential/conv1d/BiasAdd(1=
ףpia@9=
ףpia@A=
ףpia@I=
ףpia@a=����{?iǎ|W���?�Unknown
�
HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(19��v��c@99��v��c@A���x��\@I���x��\@a��0mw?i��V����?�Unknown
tHostMax"#sequential/global_max_pooling1d/Max(1�/�$�[@9�/�$�[@A�/�$�[@I�/�$�[@al&��[v?i��\s��?�Unknown
�HostRealDiv"5gradient_tape/sequential/global_max_pooling1d/truediv(1���x�Y@9���x�Y@A���x�Y@I���x�Y@a 	�i�t?i�G�<�?�Unknown
�HostResourceGather"%sequential/embedding/embedding_lookup(1�$��kY@9�$��kY@A�$��kY@I�$��kY@a9�	A;et?i����e�?�Unknown
�HostMul"1gradient_tape/sequential/global_max_pooling1d/mul(1��v���U@9��v���U@A��v���U@I��v���U@a}J����q?i-9t����?�Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1��ʡEVU@9��ʡEVU@A��ʡEVU@I��ʡEVU@aZ�ȋq?i�L����?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1H�z�/U@9H�z�/U@AH�z�/U@IH�z�/U@a�<���p?it�B����?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(1�l����T@9�l����T@A�l����T@I�l����T@a6P-Z8�p?i�k	��?�Unknown
hHostRelu"sequential/conv1d/Relu(1H�z��S@9H�z��S@AH�z��S@IH�z��S@a��Y#�o?i��P���?�Unknown
�HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333�Q@933333�Q@A33333�Q@I33333�Q@aE����sl?i��Io*�?�Unknown
�HostSum"1gradient_tape/sequential/global_max_pooling1d/Sum(1�C�l�cP@9�C�l�cP@A�C�l�cP@I�C�l�cP@a"]Y Mj?i�
�oTD�?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1#��~jP@9#��~jP@A#��~jP@I#��~jP@ak�s:s�i?i;~��!^�?�Unknown�
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1}?5^�)W@9}?5^�)W@A33333N@I33333N@aBiA�vh?i���Y6v�?�Unknown
kHostUnique"Adam/Adam/update/Unique(1� �rhK@9� �rhK@A� �rhK@I� �rhK@aIw�q��e?i���?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�z�G�A@9�z�G�A@A�z�G�A@I�z�G�A@a_r���0\?i�
�S��?�Unknown
�HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1�G�z^@@9�G�z^@@A�G�z^@@I�G�z^@@a/���CZ?i7��'(��?�Unknown
�HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1��S�?@9��S�?@A��S�?@I��S�?@a����y�X?i7��䢳�?�Unknown
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�K7�AP@@9�K7�AP@@A�G�zT=@I�G�zT=@a��G�W?i�p�g��?�Unknown
�HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1�x�&1�<@9�x�&1�<@A�x�&1�<@I�x�&1�<@a�u�_�V?iI�l8���?�Unknown
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1'1�z<@9'1�z<@A'1�z<@I'1�z<@aM���V?ipS�E��?�Unknown
dHostDataset"Iterator::Model(1%��C�C@9%��C�C@AT㥛��7@IT㥛��7@a#-��S?i��&���?�Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1�$���7@9�$���7@A�$���7@I�$���7@an�>n^�R?iq���H��?�Unknown
i HostWriteSummary"WriteSummary(1#��~j�6@9#��~j�6@A#��~j�6@I#��~j�6@a�"�8$R?i}�7�Z��?�Unknown�
�!HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(15^�IB5@95^�IB5@A5^�IB5@I5^�IB5@a�<��QQ?i�#���?�Unknown
�"HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1)\����3@9)\����3@A)\����3@I)\����3@a*Jb���O?i������?�Unknown
{#HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1�����Y2@9�����Y2@A�����Y2@I�����Y2@aIrM?i@�,�.
�?�Unknown
�$HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate(1j�t��2@9j�t��2@A��K7�2@I��K7�2@a7����L?iAÍ�g�?�Unknown
s%HostDataset"Iterator::Model::ParallelMapV2(1�(\��50@9�(\��50@A�(\��50@I�(\��50@a0I���J?iSd�m��?�Unknown
a&HostCast"sequential/Cast(1o���!0@9o���!0@Ao���!0@Io���!0@a<H#���I?i%-;(a�?�Unknown
}'HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1��ʡE�.@9��ʡE�.@A��ʡE�.@I��ʡE�.@a��J|#�H?i�?1�$�?�Unknown
�(HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1333333.@9333333.@A333333.@I333333.@a�v���:H?i�=��*�?�Unknown
g)HostMul"Adam/Adam/update/mul_2(1�/�$.@9�/�$.@A�/�$.@I�/�$.@a A<g�H?i�d��0�?�Unknown
g*HostSqrt"Adam/Adam/update/Sqrt(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a8��3�G?i�yX1�6�?�Unknown
y+HostMatMul"%gradient_tape/sequential/dense/MatMul(1���S�+@9���S�+@A���S�+@I���S�+@aQ�I`F?i�ު7'<�?�Unknown
m,HostRealDiv"Adam/Adam/update/truediv(1R���+@9R���+@AR���+@IR���+@a8�]�F?i�F��A�?�Unknown
}-HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1��/�$+@9��/�$+@A��/�$+@I��/�$+@a���H�E?i`B� G�?�Unknown
g.HostMul"Adam/Adam/update/mul_4(1���Qx*@9���Qx*@A���Qx*@I���Qx*@aL���<E?i�oL�?�Unknown
�/HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1�/�$�)@9�/�$�)@A�/�$�)@I�/�$�)@a5��D?i��k�Q�?�Unknown
k0HostCast"sequential/embedding/Cast(1?5^�IL(@9?5^�IL(@A?5^�IL(@I?5^�IL(@aY}"��~C?ikN{V�?�Unknown
�1HostStridedSlice"Agradient_tape/sequential/embedding/embedding_lookup/strided_slice(1��Q��'@9��Q��'@A��Q��'@I��Q��'@a����&C?i����D[�?�Unknown
g2HostMul"Adam/Adam/update/mul_3(1��Q��%@9��Q��%@A��Q��%@I��Q��%@aVB!Z�XA?i5����_�?�Unknown
}3HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1;�O��n%@9;�O��n%@A;�O��n%@I;�O��n%@a^ �2A?i�B�|�c�?�Unknown
g4HostMul"Adam/Adam/update/mul_1(1��ʡE�$@9��ʡE�$@A��ʡE�$@I��ʡE�$@ar����@?i�	��h�?�Unknown
�5HostSelectV2"=gradient_tape/categorical_crossentropy/clip_by_value/SelectV2(1}?5^��$@9}?5^��$@A}?5^��$@I}?5^��$@a��''oz@?i�ӿy:l�?�Unknown
�6HostDynamicStitch";gradient_tape/sequential/global_max_pooling1d/DynamicStitch(1D�l��i$@9D�l��i$@AD�l��i$@ID�l��i$@a���`@?i㶂�Rp�?�Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_2(1!�rh�-$@9!�rh�-$@A!�rh�-$@I!�rh�-$@a�"��}0@?ilf��^t�?�Unknown
�8HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1���Mb$@9���Mb$@A���Mb$@I���Mb$@a��σ@?icZ�ex�?�Unknown
e9Host
LogicalAnd"
LogicalAnd(1Zd;�$@9Zd;�$@AZd;�$@IZd;�$@a��Z�@?iF�mFk|�?�Unknown�
�:HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate(1Zd;�O�&@9Zd;�O�&@AP��nC#@IP��nC#@a���G��>?iԶ\H��?�Unknown
t;Host_FusedMatMul"sequential/dense_1/BiasAdd(1\���(�!@9\���(�!@A\���(�!@I\���(�!@ae�B<?i�`՞Ѓ�?�Unknown
g<HostAddV2"Adam/Adam/update/add(1'1�!@9'1�!@A'1�!@I'1�!@a�9�"bq;?i��>��?�Unknown
u=HostRealDiv" categorical_crossentropy/truediv(1�O��n!@9�O��n!@A�O��n!@I�O��n!@aꛠ>e;?i�m���?�Unknown
[>HostSub"
Adam/sub_3(1
ףp=
!@9
ףp=
!@A
ףp=
!@I
ףp=
!@a�Y�W;?i�mBj��?�Unknown
r?HostTensorSliceDataset"TensorSliceDataset(1��ʡE� @9��ʡE� @A��ʡE� @I��ʡE� @a5͍,�:?i�'ԏp��?�Unknown
^@HostGatherV2"GatherV2(1!�rh�� @9!�rh�� @A!�rh�� @I!�rh�� @a�N�5�:?iҀt�Ȕ�?�Unknown
lAHostIteratorGetNext"IteratorGetNext(133333s @933333s @A33333s @I33333s @al8׋e:?i�b���?�Unknown
�BHostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1��C�,)@9��C�,)@A��Mb @I��Mb @a�/�M�9?i��P��?�Unknown
gCHostStridedSlice"strided_slice(1������@9������@A������@I������@a�E!��8?i���f��?�Unknown
eDHostMul"Adam/Adam/update/mul(1%��C�@9%��C�@A%��C�@I%��C�@a��ʡ�8?i��'w��?�Unknown
gEHostMul"Adam/Adam/update/mul_5(1X9��v@9X9��v@AX9��v@IX9��v@a�E3q8?iQjM���?�Unknown
�FHostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1m����R@9m����R@Am����R@Im����R@a(��Ӧ�6?i��D�\��?�Unknown
hGHostRandomShuffle"RandomShuffle(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a(��3_6?iz+N ��?�Unknown
�HHostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1w��/�@9w��/�@Aw��/�@Iw��/�@a}!M`�4?iM5Z���?�Unknown
{IHostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�Q���@9�Q���@A�Q���@I�Q���@aZ��13?i���|��?�Unknown
VJHostSum"Sum_2(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a���~�2?i)��Lx��?�Unknown
ZKHostArgMax"ArgMax(1+�Y@9+�Y@A+�Y@I+�Y@a19�5��2?i�I��ϳ�?�Unknown
�LHostBroadcastTo"2gradient_tape/categorical_crossentropy/BroadcastTo(1��S�@9��S�@A��S�@I��S�@a��o�M�2?i���	!��?�Unknown
�MHostConcatV2":gradient_tape/sequential/embedding/embedding_lookup/concat(1%��C@9%��C@A%��C@I%��C@a��m(}2?i����p��?�Unknown
xNHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�O��n�e@9�O��n�e@A�rh���@I�rh���@aE�����0?ijJ����?�Unknown
`OHostGatherV2"
GatherV2_1(1��C�l�@9��C�l�@A��C�l�@I��C�l�@a�����0?i��=���?�Unknown
rPHostConcatenateDataset"ConcatenateDataset(1^�I�@9^�I�@A^�I�@I^�I�@a���"�0?i�"ⶾ�?�Unknown
�QHostVariableShape"Agradient_tape/sequential/embedding/embedding_lookup/VariableShape(1V-2@9V-2@AV-2@IV-2@a!]F040?i��*h���?�Unknown
tRHostAssignAddVariableOp"AssignAddVariableOp(11�Z�@91�Z�@A1�Z�@I1�Z�@a���~�/?i'� ���?�Unknown
XSHostSlice"Slice(1��x�&1@9��x�&1@A��x�&1@I��x�&1@a�2(���.?i��(ߨ��?�Unknown
\THostArgMax"ArgMax_1(1�v��/@9�v��/@A�v��/@I�v��/@a����--?i��{��?�Unknown
XUHostEqual"Equal(1�v��/@9�v��/@A�v��/@I�v��/@a����--?iL���N��?�Unknown
�VHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1)\���(@9)\���(@A)\���(@I)\���(@a�`�� $-?i"��� ��?�Unknown
�WHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1��v��@9��v��@A��v��@I��v��@a��*v4r+?i��
����?�Unknown
�XHostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1�� �r�@9�� �r�@A�� �r�@I�� �r�@a�	���!+?i����?�Unknown
~YHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1F����x@9F����x@AF����x@IF����x@a�E@m�n*?i��i1��?�Unknown
hZHostTensorDataset"TensorDataset(1�V-@9�V-@A�V-@I�V-@a�$�6(?iߎ&c���?�Unknown
�[HostSelectV2"?gradient_tape/categorical_crossentropy/clip_by_value/SelectV2_1(1!�rh��@9!�rh��@A!�rh��@I!�rh��@a��(?i��ߔ4��?�Unknown
x\HostStridedSlice"Adam/Adam/update/strided_slice(1X9��v@9X9��v@AX9��v@IX9��v@aXB�ͣ'?i��Ѯ��?�Unknown
m]HostSum"categorical_crossentropy/Sum(1��� �r@9��� �r@A��� �r@I��� �r@a��_s��'?i��(��?�Unknown
[^HostAddV2"Adam/add(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@a�
 �&?i�wa����?�Unknown
[_HostPow"
Adam/Pow_1(1��"��~@9��"��~@A��"��~@I��"��~@a�gU�&?iFζ����?�Unknown
�`HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1�K7�A`
@9�K7�A`�?A�K7�A`
@I�K7�A`�?aQ�s	�)%?ieW>��?�Unknown
�aHostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1�rh��|	@9�rh��|	@A�rh��|	@I�rh��|	@a 9զ%s$?i�ұO���?�Unknown
�bHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1R���Q	@9R���Q	@AR���Q	@IR���Q	@a�K
��P$?ix#�Y���?�Unknown
}cHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#��~j�@9#��~j�@A#��~j�@I#��~j�@aA�<Z��#?iC�����?�Unknown
�dHostDynamicStitch"4gradient_tape/categorical_crossentropy/DynamicStitch(1����x�@9����x�@A����x�@I����x�@a�(/q/#?i͹	�:��?�Unknown
�eHostRealDiv"6gradient_tape/categorical_crossentropy/truediv/RealDiv(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a=p�h	4"?iI�^��?�Unknown
XfHostCast"Cast_1(1}?5^�I@9}?5^�I@A}?5^�I@I}?5^�I@a�3�G ?iP���b��?�Unknown
ogHostSigmoid"sequential/dense_1/Sigmoid(1h��|?5@9h��|?5@Ah��|?5@Ih��|?5@a�S�y��?i�{"Y��?�Unknown
VhHostAddN"AddN(1��v��@9��v��@A��v��@I��v��@a��k�ʧ?i�l`N��?�Unknown
oiHostReadVariableOp"Adam/ReadVariableOp(1�S㥛�@9�S㥛�@A�S㥛�@I�S㥛�@a�G@��?i��N?��?�Unknown
]jHostCast"Adam/Cast_1(1+����@9+����@A+����@I+����@a�ԿQ!�?i'��%��?�Unknown
vkHostAssignAddVariableOp"AssignAddVariableOp_4(17�A`��@97�A`��@A7�A`��@I7�A`��@a&�?i�v�
��?�Unknown
VlHostCast"Cast(1-����@9-����@A-����@I-����@a�H�B�?i��)����?�Unknown
bmHostDivNoNan"div_no_nan_1(1�I+�@9�I+�@A�I+�@I�I+�@a���a ?i=K6����?�Unknown
�nHostBroadcastTo"4gradient_tape/categorical_crossentropy/BroadcastTo_1(1����x� @9����x� @A����x� @I����x� @a�Q�(U#?i�������?�Unknown
moHostLog"categorical_crossentropy/Log(1j�t��?9j�t��?Aj�t��?Ij�t��?a��J�.?i �U�i��?�Unknown
vpHostAssignAddVariableOp"AssignAddVariableOp_3(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?aHÎg�?ivј.)��?�Unknown
�qHostMinimum".categorical_crossentropy/clip_by_value/Minimum(1�(\����?9�(\����?A�(\����?I�(\����?aY��H<?i�0����?�Unknown
�rHostRealDiv"8gradient_tape/categorical_crossentropy/truediv/RealDiv_1(1%��C��?9%��C��?A%��C��?I%��C��?a$
����?i|G�G���?�Unknown
[sHostSqrt"	Adam/Sqrt(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?ay/I_ӡ?i�A/VO��?�Unknown
atHostRealDiv"Adam/truediv(1�&1��?9�&1��?A�&1��?I�&1��?am����?i�����?�Unknown
�uHostSum"4gradient_tape/categorical_crossentropy/truediv/Sum_1(1/�$��?9/�$��?A/�$��?I/�$��?a YX�9?iV��P���?�Unknown
�vHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?a��lX�?i��`��?�Unknown
�wHostGreaterEqual"Agradient_tape/categorical_crossentropy/clip_by_value/GreaterEqual(1�������?9�������?A�������?I�������?aǂg���?iI�����?�Unknown
�xHost	LessEqual">gradient_tape/categorical_crossentropy/clip_by_value/LessEqual(1#��~j��?9#��~j��?A#��~j��?I#��~j��?a�@�a{s?i3�~����?�Unknown
�yHostRealDiv"8gradient_tape/categorical_crossentropy/truediv/RealDiv_2(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a�QPA�?i�Πa��?�Unknown
YzHostSub"Adam/sub(1�C�l���?9�C�l���?A�C�l���?I�C�l���?a�D�?i�x����?�Unknown
{{HostMaximum"&categorical_crossentropy/clip_by_value(1�G�z��?9�G�z��?A�G�z��?I�G�z��?a8�sM��?i|�{����?�Unknown
o|HostSum"categorical_crossentropy/Sum_1(1q=
ףp�?9q=
ףp�?Aq=
ףp�?Iq=
ףp�?a��-�Ii?i���P��?�Unknown
�}Host	ZerosLike"?gradient_tape/categorical_crossentropy/clip_by_value/zeros_like(1�O��n�?9�O��n�?A�O��n�?I�O��n�?a�ҽ�?i|�g����?�Unknown
[~HostSub"
Adam/sub_2(1u�V�?9u�V�?Au�V�?Iu�V�?aj��oj?i�;�Ñ��?�Unknown
XHostCast"Cast_2(1}?5^�I�?9}?5^�I�?A}?5^�I�?I}?5^�I�?a�tի|?i���-��?�Unknown
Z�HostMul"Adam/mul(1D�l����?9D�l����?AD�l����?ID�l����?a9�����?i*�!����?�Unknown
��HostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a5�+\��?i���<]��?�Unknown
n�HostMul"categorical_crossentropy/mul(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?a���ţ?i��Z���?�Unknown
|�HostSum"*categorical_crossentropy/weighted_loss/Sum(1���K7�?9���K7�?A���K7�?I���K7�?a )s|�?i.*�^���?�Unknown
u�HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?a~�F%3�?ibT?H��?�Unknown
\�HostSub"
Adam/sub_1(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a�.�T#[?i��Y!���?�Unknown
��HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1
ףp=
�?9
ףp=
�?A
ףp=
�?I
ףp=
�?a�K���?i��-.��?�Unknown
w�HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?a�?W�,?iH����?�Unknown
��HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1B`��"��?9B`��"��?AB`��"��?IB`��"��?a����?i���/��?�Unknown
z�HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(15^�I�?95^�I�?A5^�I�?I5^�I�?a�6��Z�?i�=M���?�Unknown
w�HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1-�����?9-�����?A-�����?I-�����?a�H�B�?i�OU!��?�Unknown
��HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[1]::FromTensor(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?a8B���?iш����?�Unknown
b�HostIdentity"Identity(1j�t��?9j�t��?Aj�t��?Ij�t��?a��C�	?i	�$v���?�Unknown�
��HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1��Q���?9��Q���?A��Q���?I��Q���?a�|YJ	?i�E��V��?�Unknown
��Host
Reciprocal"1gradient_tape/categorical_crossentropy/Reciprocal(1�/�$�?9�/�$�?A�/�$�?I�/�$�?aa�k�mI?i��Aĳ��?�Unknown
[�HostSlice"Slice_1(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a����}?i������?�Unknown
��HostSigmoidGrad"4gradient_tape/sequential/dense_1/Sigmoid/SigmoidGrad(1V-����?9V-����?AV-����?IV-����?aז=3�i?iӺ�bg��?�Unknown
Z�HostPow"Adam/Pow(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?a}�L�[�?i�9Ⱦ��?�Unknown
n�HostNeg"categorical_crossentropy/Neg(1+���?9+���?A+���?I+���?aL4�|�?iC,���?�Unknown
��HostNeg"2gradient_tape/categorical_crossentropy/truediv/Neg(1��n���?9��n���?A��n���?I��n���?a@ѿ��v?iB�>�j��?�Unknown
w�HostAssignAddVariableOp"AssignAddVariableOp_1(1�I+��?9�I+��?A�I+��?I�I+��?a$�[m�H?i�|H���?�Unknown
a�HostDivNoNan"
div_no_nan(1�/�$�?9�/�$�?A�/�$�?I�/�$�?af�*��?i]ͧ?��?�Unknown
��Host	ZerosLike"Agradient_tape/categorical_crossentropy/clip_by_value/zeros_like_1(1V-����?9V-����?AV-����?IV-����?a�w�#_4?iO]$]��?�Unknown
��HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor(1�z�G��?9�z�G��?A�z�G��?I�z�G��?aU6�y[?i�����?�Unknown
��HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1�n����?9�n����?A�n����?I�n����?a��d_ |?i/
o���?�Unknown
��HostSum".gradient_tape/categorical_crossentropy/mul/Sum(1Zd;�O�?9Zd;�O�?AZd;�O�?IZd;�O�?a���>i?i���0��?�Unknown
��HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[0]::TensorSlice(1���S��?9���S��?A���S��?I���S��?a�/�FD��>i[�7=j��?�Unknown
��HostMul"2gradient_tape/categorical_crossentropy/truediv/mul(1q=
ףp�?9q=
ףp�?Aq=
ףp�?Iq=
ףp�?aH�V=;��>i�5���?�Unknown
U�HostMul"Mul(1�E�����?9�E�����?A�E�����?I�E�����?a��x�>i/Y�:���?�Unknown
��HostMul".gradient_tape/categorical_crossentropy/mul/Mul(1�������?9�������?A�������?I�������?a�j��R��>iB&D%��?�Unknown
��HostDivNoNan",categorical_crossentropy/weighted_loss/value(1-�����?9-�����?A-�����?I-�����?a��c(I�>i
wt�:��?�Unknown
x�HostReadVariableOp"div_no_nan/ReadVariableOp_1(1����S�?9����S�?A����S�?I����S�?a�#����>iB���f��?�Unknown
x�HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�z�G��?9�z�G��?A�z�G��?I�z�G��?aQU���>i�������?�Unknown
|�HostMul"*gradient_tape/categorical_crossentropy/mul(1�Zd;��?9�Zd;��?A�Zd;��?I�Zd;��?a�5r�9'�>i��-���?�Unknown
v�HostReadVariableOp"div_no_nan/ReadVariableOp(1X9��v��?9X9��v��?AX9��v��?IX9��v��?añ]}��>i�����?�Unknown
|�HostNeg"*gradient_tape/categorical_crossentropy/Neg(1��Q��?9��Q��?A��Q��?I��Q��?aw9����>i�������?�Unknown*��
�HostConv2DBackpropFilter";gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropFilter(1%��?�@9%��?�@A%��?�@I%��?�@a���8�.�?i���8�.�?�Unknown
lHostConv2D"sequential/conv1d/conv1d(1��/�@9��/�@A��/�@I��/�@a$N�4�?iF0�6Y�?�Unknown
�HostConv2DBackpropInput":gradient_tape/sequential/conv1d/conv1d/Conv2DBackpropInput(1��K�e�@9��K�e�@A��K�e�@I��K�e�@a����?i��6���?�Unknown
�HostCast"2gradient_tape/sequential/global_max_pooling1d/Cast(1     �v@9     �v@A     �v@I     �v@aو���U�?iΉt�i��?�Unknown
�HostEqual"3gradient_tape/sequential/global_max_pooling1d/Equal(1�Zd;q@9�Zd;q@A�Zd;q@I�Zd;q@aQND�t��?iA�yݸ�?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1��Q��m@9��Q��m@A��Q��m@I��Q��m@a����q\�?i�@��[�?�Unknown
~HostReluGrad"(gradient_tape/sequential/conv1d/ReluGrad(19��v�m@99��v�m@A9��v�m@I9��v�m@a��.خ�?i{����?�Unknown
nHostBiasAdd"sequential/conv1d/BiasAdd(1=
ףpia@9=
ףpia@A=
ףpia@I=
ףpia@aɨm�3�?il�N'\�?�Unknown
�	HostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(19��v��c@99��v��c@A���x��\@I���x��\@a	7��v�?i�x�(���?�Unknown
t
HostMax"#sequential/global_max_pooling1d/Max(1�/�$�[@9�/�$�[@A�/�$�[@I�/�$�[@a����N;�?i1�
d���?�Unknown
�HostRealDiv"5gradient_tape/sequential/global_max_pooling1d/truediv(1���x�Y@9���x�Y@A���x�Y@I���x�Y@a��\V��?i$�z�S?�?�Unknown
�HostResourceGather"%sequential/embedding/embedding_lookup(1�$��kY@9�$��kY@A�$��kY@I�$��kY@a�4gꊁ?iL�g��?�Unknown
�HostMul"1gradient_tape/sequential/global_max_pooling1d/mul(1��v���U@9��v���U@A��v���U@I��v���U@a@12E�J~?i���)��?�Unknown
HostAssignVariableOp"!Adam/Adam/update/AssignVariableOp(1��ʡEVU@9��ʡEVU@A��ʡEVU@I��ʡEVU@aݍ~�s}?i��NS���?�Unknown
�HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1H�z�/U@9H�z�/U@AH�z�/U@IH�z�/U@aJb���=}?i�dF�v7�?�Unknown
�HostBiasAddGrad"3gradient_tape/sequential/conv1d/BiasAdd/BiasAddGrad(1�l����T@9�l����T@A�l����T@I�l����T@a��E�n|?i�K�{Tp�?�Unknown
hHostRelu"sequential/conv1d/Relu(1H�z��S@9H�z��S@AH�z��S@IH�z��S@a��+� {?ig�(���?�Unknown
�HostUnsortedSegmentSum"#Adam/Adam/update/UnsortedSegmentSum(133333�Q@933333�Q@A33333�Q@I33333�Q@a�f�ayx?i5J� ���?�Unknown
�HostSum"1gradient_tape/sequential/global_max_pooling1d/Sum(1�C�l�cP@9�C�l�cP@A�C�l�cP@I�C�l�cP@a_�"U:�v?iƏ����?�Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1#��~jP@9#��~jP@A#��~jP@I#��~jP@a�}⊄1v?i�T��)1�?�Unknown�
�HostDataset">Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map(1}?5^�)W@9}?5^�)W@A33333N@I33333N@a�f
�6�t?i�i>�Z�?�Unknown
kHostUnique"Adam/Adam/update/Unique(1� �rhK@9� �rhK@A� �rhK@I� �rhK@a"���r?i�����?�Unknown
oHost_FusedMatMul"sequential/dense/Relu(1�z�G�A@9�z�G�A@A�z�G�A@I�z�G�A@aM�0?h?iޙRF1��?�Unknown
�HostAssignVariableOp"#Adam/Adam/update/AssignVariableOp_1(1�G�z^@@9�G�z^@@A�G�z^@@I�G�z^@@am�[�0�f?i���vȮ�?�Unknown
�HostAssignSubVariableOp"$Adam/Adam/update/AssignSubVariableOp(1��S�?@9��S�?@A��S�?@I��S�?@a�bҦ�we?i)�}7@��?�Unknown
�HostDataset"LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat(1�K7�AP@@9�K7�AP@@A�G�zT=@I�G�zT=@anGc��=d?ip+�}��?�Unknown
�HostResourceScatterAdd"%Adam/Adam/update/ResourceScatterAdd_1(1�x�&1�<@9�x�&1�<@A�x�&1�<@I�x�&1�<@a�y9���c?i�d��.��?�Unknown
{HostReadVariableOp"Adam/Adam/update/ReadVariableOp(1'1�z<@9'1�z<@A'1�z<@I'1�z<@ak�*�c?i�x�t���?�Unknown
dHostDataset"Iterator::Model(1%��C�C@9%��C�C@AT㥛��7@IT㥛��7@a���)^d`?i+�9�?�Unknown
}HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_2(1�$���7@9�$���7@A�$���7@I�$���7@a�$Q:P`?i6R� �?�Unknown
iHostWriteSummary"WriteSummary(1#��~j�6@9#��~j�6@A#��~j�6@I#��~j�6@a����:5_?i�5��$0�?�Unknown�
� HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(15^�IB5@95^�IB5@A5^�IB5@I5^�IB5@a/�R+W]?i`_@�>�?�Unknown
�!HostResourceScatterAdd"#Adam/Adam/update/ResourceScatterAdd(1)\����3@9)\����3@A)\����3@I)\����3@a�G�N[?i���wL�?�Unknown
{"HostMatMul"'gradient_tape/sequential/dense/MatMul_1(1�����Y2@9�����Y2@A�����Y2@I�����Y2@a��2q�SY?ik��p!Y�?�Unknown
�#HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate(1j�t��2@9j�t��2@A��K7�2@I��K7�2@a��Tv �X?i����e�?�Unknown
s$HostDataset"Iterator::Model::ParallelMapV2(1�(\��50@9�(\��50@A�(\��50@I�(\��50@a]��_V?i@�$F�p�?�Unknown
a%HostCast"sequential/Cast(1o���!0@9o���!0@Ao���!0@Io���!0@aF4���CV?i�ɂC�{�?�Unknown
}&HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_3(1��ʡE�.@9��ʡE�.@A��ʡE�.@I��ʡE�.@a�y[��1U?i�w�(y��?�Unknown
�'HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1333333.@9333333.@A333333.@I333333.@ax�\�V�T?i&X���?�Unknown
g(HostMul"Adam/Adam/update/mul_2(1�/�$.@9�/�$.@A�/�$.@I�/�$.@ay^}�>�T?i�d��@��?�Unknown
g)HostSqrt"Adam/Adam/update/Sqrt(1�&1��-@9�&1��-@A�&1��-@I�&1��-@a��f|uoT?i����x��?�Unknown
y*HostMatMul"%gradient_tape/sequential/dense/MatMul(1���S�+@9���S�+@A���S�+@I���S�+@ag?��>S?i�!���?�Unknown
m+HostRealDiv"Adam/Adam/update/truediv(1R���+@9R���+@AR���+@IR���+@as���S?i�uz���?�Unknown
},HostReadVariableOp"!Adam/Adam/update/ReadVariableOp_1(1��/�$+@9��/�$+@A��/�$+@I��/�$+@a�K�q�R?iMx3���?�Unknown
g-HostMul"Adam/Adam/update/mul_4(1���Qx*@9���Qx*@A���Qx*@I���Qx*@au&�%_DR?i�Q�b��?�Unknown
�.HostDataset"9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch(1�/�$�)@9�/�$�)@A�/�$�)@I�/�$�)@a��Zi�Q?ih�W ��?�Unknown
k/HostCast"sequential/embedding/Cast(1?5^�IL(@9?5^�IL(@A?5^�IL(@I?5^�IL(@a� Ḧ�P?i��jb��?�Unknown
�0HostStridedSlice"Agradient_tape/sequential/embedding/embedding_lookup/strided_slice(1��Q��'@9��Q��'@A��Q��'@I��Q��'@aLN*�	yP?i����?�Unknown
g1HostMul"Adam/Adam/update/mul_3(1��Q��%@9��Q��%@A��Q��%@I��Q��%@a��(�M?i_@V���?�Unknown
}2HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1;�O��n%@9;�O��n%@A;�O��n%@I;�O��n%@a��饔M?i���y��?�Unknown
g3HostMul"Adam/Adam/update/mul_1(1��ʡE�$@9��ʡE�$@A��ʡE�$@I��ʡE�$@aH�4͔�L?i	����?�Unknown
�4HostSelectV2"=gradient_tape/categorical_crossentropy/clip_by_value/SelectV2(1}?5^��$@9}?5^��$@A}?5^��$@I}?5^��$@a����XL?i�'���?�Unknown
�5HostDynamicStitch";gradient_tape/sequential/global_max_pooling1d/DynamicStitch(1D�l��i$@9D�l��i$@AD�l��i$@ID�l��i$@ay�gR�,L?i��;���?�Unknown
v6HostAssignAddVariableOp"AssignAddVariableOp_2(1!�rh�-$@9!�rh�-$@A!�rh�-$@I!�rh�-$@a��O��K?i륏U��?�Unknown
�7HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1���Mb$@9���Mb$@A���Mb$@I���Mb$@a���I�K?iOj����?�Unknown
e8Host
LogicalAnd"
LogicalAnd(1Zd;�$@9Zd;�$@AZd;�$@IZd;�$@a{�鶔�K?i�$%ͥ�?�Unknown�
�9HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate(1Zd;�O�&@9Zd;�O�&@AP��nC#@IP��nC#@a��m��J?i2@�GK$�?�Unknown
t:Host_FusedMatMul"sequential/dense_1/BiasAdd(1\���(�!@9\���(�!@A\���(�!@I\���(�!@a�g�4NH?i���^*�?�Unknown
g;HostAddV2"Adam/Adam/update/add(1'1�!@9'1�!@A'1�!@I'1�!@apݹ���G?im��E0�?�Unknown
u<HostRealDiv" categorical_crossentropy/truediv(1�O��n!@9�O��n!@A�O��n!@I�O��n!@a�b��G?i�`��)6�?�Unknown
[=HostSub"
Adam/sub_3(1
ףp=
!@9
ףp=
!@A
ףp=
!@I
ףp=
!@a�4�r΄G?i|[�
<�?�Unknown
r>HostTensorSliceDataset"TensorSliceDataset(1��ʡE� @9��ʡE� @A��ʡE� @I��ʡE� @a�D��G?i}j���A�?�Unknown
^?HostGatherV2"GatherV2(1!�rh�� @9!�rh�� @A!�rh�� @I!�rh�� @a�2�N�G?i�]1�G�?�Unknown
l@HostIteratorGetNext"IteratorGetNext(133333s @933333s @A33333s @I33333s @a-t��V�F?i�DG=M�?�Unknown
�AHostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1��C�,)@9��C�,)@A��Mb @I��Mb @a�i���6F?i��x��R�?�Unknown
gBHostStridedSlice"strided_slice(1������@9������@A������@I������@a�8�BE?i�AgmX�?�Unknown
eCHostMul"Adam/Adam/update/mul(1%��C�@9%��C�@A%��C�@I%��C�@a���E?i�t�t`]�?�Unknown
gDHostMul"Adam/Adam/update/mul_5(1X9��v@9X9��v@AX9��v@IX9��v@ax(���E?i��B�b�?�Unknown
�EHostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1m����R@9m����R@Am����R@Im����R@a7z�C?if��g�?�Unknown
hFHostRandomShuffle"RandomShuffle(1� �rh�@9� �rh�@A� �rh�@I� �rh�@a� �ZC?i �&�Fl�?�Unknown
�GHostReadVariableOp"(sequential/conv1d/BiasAdd/ReadVariableOp(1w��/�@9w��/�@Aw��/�@Iw��/�@ak��O�A?iTؼp�?�Unknown
{HHostMatMul"'gradient_tape/sequential/dense_1/MatMul(1�Q���@9�Q���@A�Q���@I�Q���@a+߁@?iߔ�O�t�?�Unknown
VIHostSum"Sum_2(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@aB}_3-@?i�le��x�?�Unknown
ZJHostArgMax"ArgMax(1+�Y@9+�Y@A+�Y@I+�Y@a��W�@?i�^���|�?�Unknown
�KHostBroadcastTo"2gradient_tape/categorical_crossentropy/BroadcastTo(1��S�@9��S�@A��S�@I��S�@aV�����??i��vc��?�Unknown
�LHostConcatV2":gradient_tape/sequential/embedding/embedding_lookup/concat(1%��C@9%��C@A%��C@I%��C@a�(��8�??i��*��?�Unknown
xMHostDataset"#Iterator::Model::ParallelMapV2::Zip(1�O��n�e@9�O��n�e@A�rh���@I�rh���@a�y<���<?i|�$#���?�Unknown
`NHostGatherV2"
GatherV2_1(1��C�l�@9��C�l�@A��C�l�@I��C�l�@a��<?i��e ��?�Unknown
rOHostConcatenateDataset"ConcatenateDataset(1^�I�@9^�I�@A^�I�@I^�I�@a�<>~��<?i�d<���?�Unknown
�PHostVariableShape"Agradient_tape/sequential/embedding/embedding_lookup/VariableShape(1V-2@9V-2@AV-2@IV-2@a��`y��;?i��:-��?�Unknown
tQHostAssignAddVariableOp"AssignAddVariableOp(11�Z�@91�Z�@A1�Z�@I1�Z�@a����t;?iM�ʛ��?�Unknown
XRHostSlice"Slice(1��x�&1@9��x�&1@A��x�&1@I��x�&1@ag\�:.}:?iwOp��?�Unknown
\SHostArgMax"ArgMax_1(1�v��/@9�v��/@A�v��/@I�v��/@a��;,9?i������?�Unknown
XTHostEqual"Equal(1�v��/@9�v��/@A�v��/@I�v��/@a��;,9?i���1��?�Unknown
�UHostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1)\���(@9)\���(@A)\���(@I)\���(@a�
\J�9?i����S��?�Unknown
�VHostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1��v��@9��v��@A��v��@I��v��@a���wk�7?i��R0G��?�Unknown
�WHostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1�� �r�@9�� �r�@A�� �r�@I�� �r�@a�*V7?i-��1��?�Unknown
~XHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1F����x@9F����x@AF����x@IF����x@a-U���6?iذBy	��?�Unknown
hYHostTensorDataset"TensorDataset(1�V-@9�V-@A�V-@I�V-@aa�lZ�4?iv�mܣ��?�Unknown
�ZHostSelectV2"?gradient_tape/categorical_crossentropy/clip_by_value/SelectV2_1(1!�rh��@9!�rh��@A!�rh��@I!�rh��@a�� I�4?i*��8��?�Unknown
x[HostStridedSlice"Adam/Adam/update/strided_slice(1X9��v@9X9��v@AX9��v@IX9��v@a��OU4?i�oó�?�Unknown
m\HostSum"categorical_crossentropy/Sum(1��� �r@9��� �r@A��� �r@I��� �r@aL��M|R4?i��M��?�Unknown
[]HostAddV2"Adam/add(1Zd;�O�@9Zd;�O�@AZd;�O�@IZd;�O�@a!,�s�3?il(�/���?�Unknown
[^HostPow"
Adam/Pow_1(1��"��~@9��"��~@A��"��~@I��"��~@a�S����2?i�` d��?�Unknown
�_HostDataset"SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range(1�K7�A`
@9�K7�A`�?A�K7�A`
@I�K7�A`�?a�h��32?i�!��S��?�Unknown
�`HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1�rh��|	@9�rh��|	@A�rh��|	@I�rh��|	@aT�Y��1?i�븆��?�Unknown
�aHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1R���Q	@9R���Q	@AR���Q	@IR���Q	@a�P�l4y1?ih&yߵ��?�Unknown
}bHostReluGrad"'gradient_tape/sequential/dense/ReluGrad(1#��~j�@9#��~j�@A#��~j�@I#��~j�@a'��1?i��y ���?�Unknown
�cHostDynamicStitch"4gradient_tape/categorical_crossentropy/DynamicStitch(1����x�@9����x�@A����x�@I����x�@a�0�5u�0?i�� /���?�Unknown
�dHostRealDiv"6gradient_tape/categorical_crossentropy/truediv/RealDiv(1㥛� �@9㥛� �@A㥛� �@I㥛� �@a(loP/?i�_6���?�Unknown
XeHostCast"Cast_1(1}?5^�I@9}?5^�I@A}?5^�I@I}?5^�I@aT�0o ,?i�h
=���?�Unknown
ofHostSigmoid"sequential/dense_1/Sigmoid(1h��|?5@9h��|?5@Ah��|?5@Ih��|?5@a�E6|Ղ*?iG,bjE��?�Unknown
VgHostAddN"AddN(1��v��@9��v��@A��v��@I��v��@ag�T^*?i�m�K���?�Unknown
ohHostReadVariableOp"Adam/ReadVariableOp(1�S㥛�@9�S㥛�@A�S㥛�@I�S㥛�@aݱ��]�)?i@�����?�Unknown
]iHostCast"Adam/Cast_1(1+����@9+����@A+����@I+����@a&];��(?i�ڌ��?�Unknown
vjHostAssignAddVariableOp"AssignAddVariableOp_4(17�A`��@97�A`��@A7�A`��@I7�A`��@aU�}O��(?i�����?�Unknown
VkHostCast"Cast(1-����@9-����@A-����@I-����@a�'^,܈(?i�~��(��?�Unknown
blHostDivNoNan"div_no_nan_1(1�I+�@9�I+�@A�I+�@I�I+�@a(���;1(?i�>����?�Unknown
�mHostBroadcastTo"4gradient_tape/categorical_crossentropy/BroadcastTo_1(1����x� @9����x� @A����x� @I����x� @a*��h�W'?i���!��?�Unknown
mnHostLog"categorical_crossentropy/Log(1j�t��?9j�t��?Aj�t��?Ij�t��?ad-ն$?i��Մl��?�Unknown
voHostAssignAddVariableOp"AssignAddVariableOp_3(1ףp=
��?9ףp=
��?Aףp=
��?Iףp=
��?a�M-��$?i��� ���?�Unknown
�pHostMinimum".categorical_crossentropy/clip_by_value/Minimum(1�(\����?9�(\����?A�(\����?I�(\����?aL�E�#?i�c����?�Unknown
�qHostRealDiv"8gradient_tape/categorical_crossentropy/truediv/RealDiv_1(1%��C��?9%��C��?A%��C��?I%��C��?a6q�Zǲ#?i�y�0��?�Unknown
[rHostSqrt"	Adam/Sqrt(1h��|?5�?9h��|?5�?Ah��|?5�?Ih��|?5�?a�]�-kw#?i��+hh��?�Unknown
asHostRealDiv"Adam/truediv(1�&1��?9�&1��?A�&1��?I�&1��?a���iuf#?i��Ϟ��?�Unknown
�tHostSum"4gradient_tape/categorical_crossentropy/truediv/Sum_1(1/�$��?9/�$��?A/�$��?I/�$��?a�M��"?i_�;����?�Unknown
�uHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1L7�A`��?9L7�A`��?AL7�A`��?IL7�A`��?aQ�U���"?i�KWz���?�Unknown
�vHostGreaterEqual"Agradient_tape/categorical_crossentropy/clip_by_value/GreaterEqual(1�������?9�������?A�������?I�������?a�=���~"?i#�e��?�Unknown
�wHost	LessEqual">gradient_tape/categorical_crossentropy/clip_by_value/LessEqual(1#��~j��?9#��~j��?A#��~j��?I#��~j��?akt]s"?i���F��?�Unknown
�xHostRealDiv"8gradient_tape/categorical_crossentropy/truediv/RealDiv_2(1��K7�A�?9��K7�A�?A��K7�A�?I��K7�A�?a;�W��"?i��h��?�Unknown
YyHostSub"Adam/sub(1�C�l���?9�C�l���?A�C�l���?I�C�l���?a�|����!?i�w'l���?�Unknown
{zHostMaximum"&categorical_crossentropy/clip_by_value(1�G�z��?9�G�z��?A�G�z��?I�G�z��?aS�
θ!?i#%����?�Unknown
o{HostSum"categorical_crossentropy/Sum_1(1q=
ףp�?9q=
ףp�?Aq=
ףp�?Iq=
ףp�?a&�y�g�!?i�<�߻��?�Unknown
�|Host	ZerosLike"?gradient_tape/categorical_crossentropy/clip_by_value/zeros_like(1�O��n�?9�O��n�?A�O��n�?I�O��n�?al�3dM!?isnŵ���?�Unknown
[}HostSub"
Adam/sub_2(1u�V�?9u�V�?Au�V�?Iu�V�?a�
{��J!?i$��^���?�Unknown
X~HostCast"Cast_2(1}?5^�I�?9}?5^�I�?A}?5^�I�?I}?5^�I�?a�'}t�� ?i�������?�Unknown
YHostMul"Adam/mul(1D�l����?9D�l����?AD�l����?ID�l����?ap���T ?i�E����?�Unknown
��HostReadVariableOp"4sequential/conv1d/conv1d/ExpandDims_1/ReadVariableOp(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a��޺�N ?i�M!����?�Unknown
n�HostMul"categorical_crossentropy/mul(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?a�@S ?i�RL���?�Unknown
|�HostSum"*categorical_crossentropy/weighted_loss/Sum(1���K7�?9���K7�?A���K7�?I���K7�?a�j ?i֫H����?�Unknown
u�HostReadVariableOp"Adam/Cast/ReadVariableOp(1333333�?9333333�?A333333�?I333333�?aY& ʫ ?i�K����?�Unknown
\�HostSub"
Adam/sub_1(1o��ʡ�?9o��ʡ�?Ao��ʡ�?Io��ʡ�?a.P�?i�����?�Unknown
��HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1
ףp=
�?9
ףp=
�?A
ףp=
�?I
ףp=
�?a^0L+$
?i�r�����?�Unknown
w�HostReadVariableOp"Adam/Cast_3/ReadVariableOp(1)\���(�?9)\���(�?A)\���(�?I)\���(�?ac�&5�?i	�T����?�Unknown
��HostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1B`��"��?9B`��"��?AB`��"��?IB`��"��?a´�N�g?i��ҍ��?�Unknown
z�HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(15^�I�?95^�I�?A5^�I�?I5^�I�?a�_��*<?iZ��_��?�Unknown
w�HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1-�����?9-�����?A-�����?I-�����?a�'^,܈?iK���#��?�Unknown
��HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[1]::FromTensor(1�K7�A`�?9�K7�A`�?A�K7�A`�?I�K7�A`�?aX[ ͆�?iNN����?�Unknown
b�HostIdentity"Identity(1j�t��?9j�t��?Aj�t��?Ij�t��?a���"�?i�e����?�Unknown�
��HostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1��Q���?9��Q���?A��Q���?I��Q���?av@���?i��^�B��?�Unknown
��Host
Reciprocal"1gradient_tape/categorical_crossentropy/Reciprocal(1�/�$�?9�/�$�?A�/�$�?I�/�$�?ap-�?iXH ����?�Unknown
[�HostSlice"Slice_1(1�x�&1�?9�x�&1�?A�x�&1�?I�x�&1�?a�ٲFSX?i�}��}��?�Unknown
��HostSigmoidGrad"4gradient_tape/sequential/dense_1/Sigmoid/SigmoidGrad(1V-����?9V-����?AV-����?IV-����?a��]G?i������?�Unknown
Z�HostPow"Adam/Pow(1�Zd;�?9�Zd;�?A�Zd;�?I�Zd;�?a�u���?i0�uE���?�Unknown
n�HostNeg"categorical_crossentropy/Neg(1+���?9+���?A+���?I+���?a"v�&�?iᔬ~B��?�Unknown
��HostNeg"2gradient_tape/categorical_crossentropy/truediv/Neg(1��n���?9��n���?A��n���?I��n���?a�_�1v?i�:50���?�Unknown
w�HostAssignAddVariableOp"AssignAddVariableOp_1(1�I+��?9�I+��?A�I+��?I�I+��?a���K�N?iL�'�h��?�Unknown
a�HostDivNoNan"
div_no_nan(1�/�$�?9�/�$�?A�/�$�?I�/�$�?a>!;Q�D?i%$r����?�Unknown
��Host	ZerosLike"Agradient_tape/categorical_crossentropy/clip_by_value/zeros_like_1(1V-����?9V-����?AV-����?IV-����?a�����?iZ�v��?�Unknown
��HostDataset"NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[4]::Concatenate[1]::FromTensor(1�z�G��?9�z�G��?A�z�G��?I�z�G��?a�{F�?iF!C���?�Unknown
��HostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1�n����?9�n����?A�n����?I�n����?a�&�?i>ޅ�m��?�Unknown
��HostSum".gradient_tape/categorical_crossentropy/mul/Sum(1Zd;�O�?9Zd;�O�?AZd;�O�?IZd;�O�?a���?j?ii��;���?�Unknown
��HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[5]::Concatenate[0]::TensorSlice(1���S��?9���S��?A���S��?I���S��?a��n�?iߴ?�E��?�Unknown
��HostMul"2gradient_tape/categorical_crossentropy/truediv/mul(1q=
ףp�?9q=
ףp�?Aq=
ףp�?Iq=
ףp�?a) �#?i_��E���?�Unknown
U�HostMul"Mul(1�E�����?9�E�����?A�E�����?I�E�����?a�Y�"P;?i�3��?�Unknown
��HostMul".gradient_tape/categorical_crossentropy/mul/Mul(1�������?9�������?A�������?I�������?a��#�0?iz�_��?�Unknown
��HostDivNoNan",categorical_crossentropy/weighted_loss/value(1-�����?9-�����?A-�����?I-�����?aO��<+?iH�z����?�Unknown
x�HostReadVariableOp"div_no_nan/ReadVariableOp_1(1����S�?9����S�?A����S�?I����S�?a��4���?i�I���?�Unknown
x�HostReadVariableOp"div_no_nan_1/ReadVariableOp(1�z�G��?9�z�G��?A�z�G��?I�z�G��?a��Ό?i��BB��?�Unknown
|�HostMul"*gradient_tape/categorical_crossentropy/mul(1�Zd;��?9�Zd;��?A�Zd;��?I�Zd;��?a�L>$dy ?i�(���?�Unknown
v�HostReadVariableOp"div_no_nan/ReadVariableOp(1X9��v��?9X9��v��?AX9��v��?IX9��v��?a�>�b ?i�/����?�Unknown
|�HostNeg"*gradient_tape/categorical_crossentropy/Neg(1��Q��?9��Q��?A��Q��?I��Q��?a���qh&�>i�������?�Unknown