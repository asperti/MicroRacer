??
??
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28??
?
perturbable1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameperturbable1/kernel
{
'perturbable1/kernel/Read/ReadVariableOpReadVariableOpperturbable1/kernel*
_output_shapes

:@*
dtype0
z
perturbable1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameperturbable1/bias
s
%perturbable1/bias/Read/ReadVariableOpReadVariableOpperturbable1/bias*
_output_shapes
:@*
dtype0
?
layer_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@**
shared_namelayer_normalization/gamma
?
-layer_normalization/gamma/Read/ReadVariableOpReadVariableOplayer_normalization/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namelayer_normalization/beta
?
,layer_normalization/beta/Read/ReadVariableOpReadVariableOplayer_normalization/beta*
_output_shapes
:@*
dtype0
?
perturbable2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@@*$
shared_nameperturbable2/kernel
{
'perturbable2/kernel/Read/ReadVariableOpReadVariableOpperturbable2/kernel*
_output_shapes

:@@*
dtype0
z
perturbable2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameperturbable2/bias
s
%perturbable2/bias/Read/ReadVariableOpReadVariableOpperturbable2/bias*
_output_shapes
:@*
dtype0
?
layer_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namelayer_normalization_1/gamma
?
/layer_normalization_1/gamma/Read/ReadVariableOpReadVariableOplayer_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
layer_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namelayer_normalization_1/beta
?
.layer_normalization_1/beta/Read/ReadVariableOpReadVariableOplayer_normalization_1/beta*
_output_shapes
:@*
dtype0
?
perturbable3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*$
shared_nameperturbable3/kernel
{
'perturbable3/kernel/Read/ReadVariableOpReadVariableOpperturbable3/kernel*
_output_shapes

:@*
dtype0
z
perturbable3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameperturbable3/bias
s
%perturbable3/bias/Read/ReadVariableOpReadVariableOpperturbable3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
 
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
q
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
q
axis
	 gamma
!beta
"	variables
#trainable_variables
$regularization_losses
%	keras_api
h

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
F
0
1
2
3
4
5
 6
!7
&8
'9
F
0
1
2
3
4
5
 6
!7
&8
'9
 
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
	regularization_losses
 
_]
VARIABLE_VALUEperturbable1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEperturbable1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
 
db
VARIABLE_VALUElayer_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
_]
VARIABLE_VALUEperturbable2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEperturbable2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
 
fd
VARIABLE_VALUElayer_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUElayer_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
"	variables
#trainable_variables
$regularization_losses
_]
VARIABLE_VALUEperturbable3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEperturbable3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

&0
'1

&0
'1
 
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
 
*
0
1
2
3
4
5
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1perturbable1/kernelperturbable1/biaslayer_normalization/gammalayer_normalization/betaperturbable2/kernelperturbable2/biaslayer_normalization_1/gammalayer_normalization_1/betaperturbable3/kernelperturbable3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_25142114
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'perturbable1/kernel/Read/ReadVariableOp%perturbable1/bias/Read/ReadVariableOp-layer_normalization/gamma/Read/ReadVariableOp,layer_normalization/beta/Read/ReadVariableOp'perturbable2/kernel/Read/ReadVariableOp%perturbable2/bias/Read/ReadVariableOp/layer_normalization_1/gamma/Read/ReadVariableOp.layer_normalization_1/beta/Read/ReadVariableOp'perturbable3/kernel/Read/ReadVariableOp%perturbable3/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_25142581
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameperturbable1/kernelperturbable1/biaslayer_normalization/gammalayer_normalization/betaperturbable2/kernelperturbable2/biaslayer_normalization_1/gammalayer_normalization_1/betaperturbable3/kernelperturbable3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_25142621??
?
?
/__inference_perturbable2_layer_call_fn_25142446

inputs
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
J__inference_perturbable1_layer_call_and_return_conditional_losses_25142386

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?z
?	
#__inference__wrapped_model_25141697
input_1C
1actor_perturbable1_matmul_readvariableop_resource:@@
2actor_perturbable1_biasadd_readvariableop_resource:@E
7actor_layer_normalization_mul_2_readvariableop_resource:@C
5actor_layer_normalization_add_readvariableop_resource:@C
1actor_perturbable2_matmul_readvariableop_resource:@@@
2actor_perturbable2_biasadd_readvariableop_resource:@G
9actor_layer_normalization_1_mul_2_readvariableop_resource:@E
7actor_layer_normalization_1_add_readvariableop_resource:@C
1actor_perturbable3_matmul_readvariableop_resource:@@
2actor_perturbable3_biasadd_readvariableop_resource:
identity??,actor/layer_normalization/add/ReadVariableOp?.actor/layer_normalization/mul_2/ReadVariableOp?.actor/layer_normalization_1/add/ReadVariableOp?0actor/layer_normalization_1/mul_2/ReadVariableOp?)actor/perturbable1/BiasAdd/ReadVariableOp?(actor/perturbable1/MatMul/ReadVariableOp?)actor/perturbable2/BiasAdd/ReadVariableOp?(actor/perturbable2/MatMul/ReadVariableOp?)actor/perturbable3/BiasAdd/ReadVariableOp?(actor/perturbable3/MatMul/ReadVariableOp?
(actor/perturbable1/MatMul/ReadVariableOpReadVariableOp1actor_perturbable1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
actor/perturbable1/MatMulMatMulinput_10actor/perturbable1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)actor/perturbable1/BiasAdd/ReadVariableOpReadVariableOp2actor_perturbable1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
actor/perturbable1/BiasAddBiasAdd#actor/perturbable1/MatMul:product:01actor/perturbable1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
actor/perturbable1/ReluRelu#actor/perturbable1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@t
actor/layer_normalization/ShapeShape%actor/perturbable1/Relu:activations:0*
T0*
_output_shapes
:w
-actor/layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/actor/layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/actor/layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'actor/layer_normalization/strided_sliceStridedSlice(actor/layer_normalization/Shape:output:06actor/layer_normalization/strided_slice/stack:output:08actor/layer_normalization/strided_slice/stack_1:output:08actor/layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
actor/layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
actor/layer_normalization/mulMul(actor/layer_normalization/mul/x:output:00actor/layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: y
/actor/layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:{
1actor/layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1actor/layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)actor/layer_normalization/strided_slice_1StridedSlice(actor/layer_normalization/Shape:output:08actor/layer_normalization/strided_slice_1/stack:output:0:actor/layer_normalization/strided_slice_1/stack_1:output:0:actor/layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!actor/layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
actor/layer_normalization/mul_1Mul*actor/layer_normalization/mul_1/x:output:02actor/layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: k
)actor/layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :k
)actor/layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
'actor/layer_normalization/Reshape/shapePack2actor/layer_normalization/Reshape/shape/0:output:0!actor/layer_normalization/mul:z:0#actor/layer_normalization/mul_1:z:02actor/layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
!actor/layer_normalization/ReshapeReshape%actor/perturbable1/Relu:activations:00actor/layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@~
%actor/layer_normalization/ones/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:i
$actor/layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
actor/layer_normalization/onesFill.actor/layer_normalization/ones/packed:output:0-actor/layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????
&actor/layer_normalization/zeros/packedPack!actor/layer_normalization/mul:z:0*
N*
T0*
_output_shapes
:j
%actor/layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
actor/layer_normalization/zerosFill/actor/layer_normalization/zeros/packed:output:0.actor/layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????b
actor/layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB d
!actor/layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
*actor/layer_normalization/FusedBatchNormV3FusedBatchNormV3*actor/layer_normalization/Reshape:output:0'actor/layer_normalization/ones:output:0(actor/layer_normalization/zeros:output:0(actor/layer_normalization/Const:output:0*actor/layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
#actor/layer_normalization/Reshape_1Reshape.actor/layer_normalization/FusedBatchNormV3:y:0(actor/layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@?
.actor/layer_normalization/mul_2/ReadVariableOpReadVariableOp7actor_layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
actor/layer_normalization/mul_2Mul,actor/layer_normalization/Reshape_1:output:06actor/layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
,actor/layer_normalization/add/ReadVariableOpReadVariableOp5actor_layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
actor/layer_normalization/addAddV2#actor/layer_normalization/mul_2:z:04actor/layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
(actor/perturbable2/MatMul/ReadVariableOpReadVariableOp1actor_perturbable2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
actor/perturbable2/MatMulMatMul!actor/layer_normalization/add:z:00actor/perturbable2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
)actor/perturbable2/BiasAdd/ReadVariableOpReadVariableOp2actor_perturbable2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
actor/perturbable2/BiasAddBiasAdd#actor/perturbable2/MatMul:product:01actor/perturbable2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@v
actor/perturbable2/ReluRelu#actor/perturbable2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@v
!actor/layer_normalization_1/ShapeShape%actor/perturbable2/Relu:activations:0*
T0*
_output_shapes
:y
/actor/layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1actor/layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1actor/layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)actor/layer_normalization_1/strided_sliceStridedSlice*actor/layer_normalization_1/Shape:output:08actor/layer_normalization_1/strided_slice/stack:output:0:actor/layer_normalization_1/strided_slice/stack_1:output:0:actor/layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!actor/layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
actor/layer_normalization_1/mulMul*actor/layer_normalization_1/mul/x:output:02actor/layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: {
1actor/layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:}
3actor/layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3actor/layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+actor/layer_normalization_1/strided_slice_1StridedSlice*actor/layer_normalization_1/Shape:output:0:actor/layer_normalization_1/strided_slice_1/stack:output:0<actor/layer_normalization_1/strided_slice_1/stack_1:output:0<actor/layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#actor/layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
!actor/layer_normalization_1/mul_1Mul,actor/layer_normalization_1/mul_1/x:output:04actor/layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: m
+actor/layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :m
+actor/layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
)actor/layer_normalization_1/Reshape/shapePack4actor/layer_normalization_1/Reshape/shape/0:output:0#actor/layer_normalization_1/mul:z:0%actor/layer_normalization_1/mul_1:z:04actor/layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
#actor/layer_normalization_1/ReshapeReshape%actor/perturbable2/Relu:activations:02actor/layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@?
'actor/layer_normalization_1/ones/packedPack#actor/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:k
&actor/layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
 actor/layer_normalization_1/onesFill0actor/layer_normalization_1/ones/packed:output:0/actor/layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:??????????
(actor/layer_normalization_1/zeros/packedPack#actor/layer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:l
'actor/layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
!actor/layer_normalization_1/zerosFill1actor/layer_normalization_1/zeros/packed:output:00actor/layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????d
!actor/layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB f
#actor/layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
,actor/layer_normalization_1/FusedBatchNormV3FusedBatchNormV3,actor/layer_normalization_1/Reshape:output:0)actor/layer_normalization_1/ones:output:0*actor/layer_normalization_1/zeros:output:0*actor/layer_normalization_1/Const:output:0,actor/layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
%actor/layer_normalization_1/Reshape_1Reshape0actor/layer_normalization_1/FusedBatchNormV3:y:0*actor/layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:?????????@?
0actor/layer_normalization_1/mul_2/ReadVariableOpReadVariableOp9actor_layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
!actor/layer_normalization_1/mul_2Mul.actor/layer_normalization_1/Reshape_1:output:08actor/layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
.actor/layer_normalization_1/add/ReadVariableOpReadVariableOp7actor_layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
actor/layer_normalization_1/addAddV2%actor/layer_normalization_1/mul_2:z:06actor/layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
(actor/perturbable3/MatMul/ReadVariableOpReadVariableOp1actor_perturbable3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
actor/perturbable3/MatMulMatMul#actor/layer_normalization_1/add:z:00actor/perturbable3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
)actor/perturbable3/BiasAdd/ReadVariableOpReadVariableOp2actor_perturbable3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
actor/perturbable3/BiasAddBiasAdd#actor/perturbable3/MatMul:product:01actor/perturbable3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????v
actor/perturbable3/TanhTanh#actor/perturbable3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????j
IdentityIdentityactor/perturbable3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp-^actor/layer_normalization/add/ReadVariableOp/^actor/layer_normalization/mul_2/ReadVariableOp/^actor/layer_normalization_1/add/ReadVariableOp1^actor/layer_normalization_1/mul_2/ReadVariableOp*^actor/perturbable1/BiasAdd/ReadVariableOp)^actor/perturbable1/MatMul/ReadVariableOp*^actor/perturbable2/BiasAdd/ReadVariableOp)^actor/perturbable2/MatMul/ReadVariableOp*^actor/perturbable3/BiasAdd/ReadVariableOp)^actor/perturbable3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2\
,actor/layer_normalization/add/ReadVariableOp,actor/layer_normalization/add/ReadVariableOp2`
.actor/layer_normalization/mul_2/ReadVariableOp.actor/layer_normalization/mul_2/ReadVariableOp2`
.actor/layer_normalization_1/add/ReadVariableOp.actor/layer_normalization_1/add/ReadVariableOp2d
0actor/layer_normalization_1/mul_2/ReadVariableOp0actor/layer_normalization_1/mul_2/ReadVariableOp2V
)actor/perturbable1/BiasAdd/ReadVariableOp)actor/perturbable1/BiasAdd/ReadVariableOp2T
(actor/perturbable1/MatMul/ReadVariableOp(actor/perturbable1/MatMul/ReadVariableOp2V
)actor/perturbable2/BiasAdd/ReadVariableOp)actor/perturbable2/BiasAdd/ReadVariableOp2T
(actor/perturbable2/MatMul/ReadVariableOp(actor/perturbable2/MatMul/ReadVariableOp2V
)actor/perturbable3/BiasAdd/ReadVariableOp)actor/perturbable3/BiasAdd/ReadVariableOp2T
(actor/perturbable3/MatMul/ReadVariableOp(actor/perturbable3/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
(__inference_actor_layer_call_fn_25142029
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_25141981o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
C__inference_actor_layer_call_and_return_conditional_losses_25142087
input_1'
perturbable1_25142061:@#
perturbable1_25142063:@*
layer_normalization_25142066:@*
layer_normalization_25142068:@'
perturbable2_25142071:@@#
perturbable2_25142073:@,
layer_normalization_1_25142076:@,
layer_normalization_1_25142078:@'
perturbable3_25142081:@#
perturbable3_25142083:
identity??+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?$perturbable1/StatefulPartitionedCall?$perturbable2/StatefulPartitionedCall?$perturbable3/StatefulPartitionedCall?
$perturbable1/StatefulPartitionedCallStatefulPartitionedCallinput_1perturbable1_25142061perturbable1_25142063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall-perturbable1/StatefulPartitionedCall:output:0layer_normalization_25142066layer_normalization_25142068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763?
$perturbable2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0perturbable2_25142071perturbable2_25142073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall-perturbable2/StatefulPartitionedCall:output:0layer_normalization_1_25142076layer_normalization_1_25142078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828?
$perturbable3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0perturbable3_25142081perturbable3_25142083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845|
IdentityIdentity-perturbable3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall%^perturbable1/StatefulPartitionedCall%^perturbable2/StatefulPartitionedCall%^perturbable3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2L
$perturbable1/StatefulPartitionedCall$perturbable1/StatefulPartitionedCall2L
$perturbable2/StatefulPartitionedCall$perturbable2/StatefulPartitionedCall2L
$perturbable3/StatefulPartitionedCall$perturbable3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_perturbable3_layer_call_fn_25142517

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_25142058
input_1'
perturbable1_25142032:@#
perturbable1_25142034:@*
layer_normalization_25142037:@*
layer_normalization_25142039:@'
perturbable2_25142042:@@#
perturbable2_25142044:@,
layer_normalization_1_25142047:@,
layer_normalization_1_25142049:@'
perturbable3_25142052:@#
perturbable3_25142054:
identity??+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?$perturbable1/StatefulPartitionedCall?$perturbable2/StatefulPartitionedCall?$perturbable3/StatefulPartitionedCall?
$perturbable1/StatefulPartitionedCallStatefulPartitionedCallinput_1perturbable1_25142032perturbable1_25142034*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall-perturbable1/StatefulPartitionedCall:output:0layer_normalization_25142037layer_normalization_25142039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763?
$perturbable2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0perturbable2_25142042perturbable2_25142044*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall-perturbable2/StatefulPartitionedCall:output:0layer_normalization_1_25142047layer_normalization_1_25142049*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828?
$perturbable3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0perturbable3_25142052perturbable3_25142054*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845|
IdentityIdentity-perturbable3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall%^perturbable1/StatefulPartitionedCall%^perturbable2/StatefulPartitionedCall%^perturbable3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2L
$perturbable1/StatefulPartitionedCall$perturbable1/StatefulPartitionedCall2L
$perturbable2/StatefulPartitionedCall$perturbable2/StatefulPartitionedCall2L
$perturbable3/StatefulPartitionedCall$perturbable3/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
(__inference_actor_layer_call_fn_25141875
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_25141852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
J__inference_perturbable3_layer_call_and_return_conditional_losses_25142528

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????P
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????W
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_25141981

inputs'
perturbable1_25141955:@#
perturbable1_25141957:@*
layer_normalization_25141960:@*
layer_normalization_25141962:@'
perturbable2_25141965:@@#
perturbable2_25141967:@,
layer_normalization_1_25141970:@,
layer_normalization_1_25141972:@'
perturbable3_25141975:@#
perturbable3_25141977:
identity??+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?$perturbable1/StatefulPartitionedCall?$perturbable2/StatefulPartitionedCall?$perturbable3/StatefulPartitionedCall?
$perturbable1/StatefulPartitionedCallStatefulPartitionedCallinputsperturbable1_25141955perturbable1_25141957*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall-perturbable1/StatefulPartitionedCall:output:0layer_normalization_25141960layer_normalization_25141962*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763?
$perturbable2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0perturbable2_25141965perturbable2_25141967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall-perturbable2/StatefulPartitionedCall:output:0layer_normalization_1_25141970layer_normalization_1_25141972*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828?
$perturbable3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0perturbable3_25141975perturbable3_25141977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845|
IdentityIdentity-perturbable3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall%^perturbable1/StatefulPartitionedCall%^perturbable2/StatefulPartitionedCall%^perturbable3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2L
$perturbable1/StatefulPartitionedCall$perturbable1/StatefulPartitionedCall2L
$perturbable2/StatefulPartitionedCall$perturbable2/StatefulPartitionedCall2L
$perturbable3/StatefulPartitionedCall$perturbable3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25142508

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?p
?
C__inference_actor_layer_call_and_return_conditional_losses_25142265

inputs=
+perturbable1_matmul_readvariableop_resource:@:
,perturbable1_biasadd_readvariableop_resource:@?
1layer_normalization_mul_2_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@=
+perturbable2_matmul_readvariableop_resource:@@:
,perturbable2_biasadd_readvariableop_resource:@A
3layer_normalization_1_mul_2_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@=
+perturbable3_matmul_readvariableop_resource:@:
,perturbable3_biasadd_readvariableop_resource:
identity??&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_2/ReadVariableOp?#perturbable1/BiasAdd/ReadVariableOp?"perturbable1/MatMul/ReadVariableOp?#perturbable2/BiasAdd/ReadVariableOp?"perturbable2/MatMul/ReadVariableOp?#perturbable3/BiasAdd/ReadVariableOp?"perturbable3/MatMul/ReadVariableOp?
"perturbable1/MatMul/ReadVariableOpReadVariableOp+perturbable1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
perturbable1/MatMulMatMulinputs*perturbable1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#perturbable1/BiasAdd/ReadVariableOpReadVariableOp,perturbable1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
perturbable1/BiasAddBiasAddperturbable1/MatMul:product:0+perturbable1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
perturbable1/ReluReluperturbable1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@h
layer_normalization/ShapeShapeperturbable1/Relu:activations:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization/ReshapeReshapeperturbable1/Relu:activations:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@r
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????s
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
"perturbable2/MatMul/ReadVariableOpReadVariableOp+perturbable2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
perturbable2/MatMulMatMullayer_normalization/add:z:0*perturbable2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#perturbable2/BiasAdd/ReadVariableOpReadVariableOp,perturbable2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
perturbable2/BiasAddBiasAddperturbable2/MatMul:product:0+perturbable2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
perturbable2/ReluReluperturbable2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@j
layer_normalization_1/ShapeShapeperturbable2/Relu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_1/ReshapeReshapeperturbable2/Relu:activations:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@v
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:?????????w
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:?????????@?
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
"perturbable3/MatMul/ReadVariableOpReadVariableOp+perturbable3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
perturbable3/MatMulMatMullayer_normalization_1/add:z:0*perturbable3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#perturbable3/BiasAdd/ReadVariableOpReadVariableOp,perturbable3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
perturbable3/BiasAddBiasAddperturbable3/MatMul:product:0+perturbable3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
perturbable3/TanhTanhperturbable3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentityperturbable3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp$^perturbable1/BiasAdd/ReadVariableOp#^perturbable1/MatMul/ReadVariableOp$^perturbable2/BiasAdd/ReadVariableOp#^perturbable2/MatMul/ReadVariableOp$^perturbable3/BiasAdd/ReadVariableOp#^perturbable3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp2J
#perturbable1/BiasAdd/ReadVariableOp#perturbable1/BiasAdd/ReadVariableOp2H
"perturbable1/MatMul/ReadVariableOp"perturbable1/MatMul/ReadVariableOp2J
#perturbable2/BiasAdd/ReadVariableOp#perturbable2/BiasAdd/ReadVariableOp2H
"perturbable2/MatMul/ReadVariableOp"perturbable2/MatMul/ReadVariableOp2J
#perturbable3/BiasAdd/ReadVariableOp#perturbable3/BiasAdd/ReadVariableOp2H
"perturbable3/MatMul/ReadVariableOp"perturbable3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?p
?
C__inference_actor_layer_call_and_return_conditional_losses_25142366

inputs=
+perturbable1_matmul_readvariableop_resource:@:
,perturbable1_biasadd_readvariableop_resource:@?
1layer_normalization_mul_2_readvariableop_resource:@=
/layer_normalization_add_readvariableop_resource:@=
+perturbable2_matmul_readvariableop_resource:@@:
,perturbable2_biasadd_readvariableop_resource:@A
3layer_normalization_1_mul_2_readvariableop_resource:@?
1layer_normalization_1_add_readvariableop_resource:@=
+perturbable3_matmul_readvariableop_resource:@:
,perturbable3_biasadd_readvariableop_resource:
identity??&layer_normalization/add/ReadVariableOp?(layer_normalization/mul_2/ReadVariableOp?(layer_normalization_1/add/ReadVariableOp?*layer_normalization_1/mul_2/ReadVariableOp?#perturbable1/BiasAdd/ReadVariableOp?"perturbable1/MatMul/ReadVariableOp?#perturbable2/BiasAdd/ReadVariableOp?"perturbable2/MatMul/ReadVariableOp?#perturbable3/BiasAdd/ReadVariableOp?"perturbable3/MatMul/ReadVariableOp?
"perturbable1/MatMul/ReadVariableOpReadVariableOp+perturbable1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
perturbable1/MatMulMatMulinputs*perturbable1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#perturbable1/BiasAdd/ReadVariableOpReadVariableOp,perturbable1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
perturbable1/BiasAddBiasAddperturbable1/MatMul:product:0+perturbable1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
perturbable1/ReluReluperturbable1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@h
layer_normalization/ShapeShapeperturbable1/Relu:activations:0*
T0*
_output_shapes
:q
'layer_normalization/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)layer_normalization/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)layer_normalization/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
!layer_normalization/strided_sliceStridedSlice"layer_normalization/Shape:output:00layer_normalization/strided_slice/stack:output:02layer_normalization/strided_slice/stack_1:output:02layer_normalization/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask[
layer_normalization/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mulMul"layer_normalization/mul/x:output:0*layer_normalization/strided_slice:output:0*
T0*
_output_shapes
: s
)layer_normalization/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization/strided_slice_1StridedSlice"layer_normalization/Shape:output:02layer_normalization/strided_slice_1/stack:output:04layer_normalization/strided_slice_1/stack_1:output:04layer_normalization/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization/mul_1Mul$layer_normalization/mul_1/x:output:0,layer_normalization/strided_slice_1:output:0*
T0*
_output_shapes
: e
#layer_normalization/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :e
#layer_normalization/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
!layer_normalization/Reshape/shapePack,layer_normalization/Reshape/shape/0:output:0layer_normalization/mul:z:0layer_normalization/mul_1:z:0,layer_normalization/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization/ReshapeReshapeperturbable1/Relu:activations:0*layer_normalization/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@r
layer_normalization/ones/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:c
layer_normalization/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization/onesFill(layer_normalization/ones/packed:output:0'layer_normalization/ones/Const:output:0*
T0*#
_output_shapes
:?????????s
 layer_normalization/zeros/packedPacklayer_normalization/mul:z:0*
N*
T0*
_output_shapes
:d
layer_normalization/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization/zerosFill)layer_normalization/zeros/packed:output:0(layer_normalization/zeros/Const:output:0*
T0*#
_output_shapes
:?????????\
layer_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB ^
layer_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
$layer_normalization/FusedBatchNormV3FusedBatchNormV3$layer_normalization/Reshape:output:0!layer_normalization/ones:output:0"layer_normalization/zeros:output:0"layer_normalization/Const:output:0$layer_normalization/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization/Reshape_1Reshape(layer_normalization/FusedBatchNormV3:y:0"layer_normalization/Shape:output:0*
T0*'
_output_shapes
:?????????@?
(layer_normalization/mul_2/ReadVariableOpReadVariableOp1layer_normalization_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization/mul_2Mul&layer_normalization/Reshape_1:output:00layer_normalization/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
&layer_normalization/add/ReadVariableOpReadVariableOp/layer_normalization_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization/addAddV2layer_normalization/mul_2:z:0.layer_normalization/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
"perturbable2/MatMul/ReadVariableOpReadVariableOp+perturbable2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0?
perturbable2/MatMulMatMullayer_normalization/add:z:0*perturbable2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
#perturbable2/BiasAdd/ReadVariableOpReadVariableOp,perturbable2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
perturbable2/BiasAddBiasAddperturbable2/MatMul:product:0+perturbable2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
perturbable2/ReluReluperturbable2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@j
layer_normalization_1/ShapeShapeperturbable2/Relu:activations:0*
T0*
_output_shapes
:s
)layer_normalization_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+layer_normalization_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+layer_normalization_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
#layer_normalization_1/strided_sliceStridedSlice$layer_normalization_1/Shape:output:02layer_normalization_1/strided_slice/stack:output:04layer_normalization_1/strided_slice/stack_1:output:04layer_normalization_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask]
layer_normalization_1/mul/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mulMul$layer_normalization_1/mul/x:output:0,layer_normalization_1/strided_slice:output:0*
T0*
_output_shapes
: u
+layer_normalization_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-layer_normalization_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
%layer_normalization_1/strided_slice_1StridedSlice$layer_normalization_1/Shape:output:04layer_normalization_1/strided_slice_1/stack:output:06layer_normalization_1/strided_slice_1/stack_1:output:06layer_normalization_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
layer_normalization_1/mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :?
layer_normalization_1/mul_1Mul&layer_normalization_1/mul_1/x:output:0.layer_normalization_1/strided_slice_1:output:0*
T0*
_output_shapes
: g
%layer_normalization_1/Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :g
%layer_normalization_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
#layer_normalization_1/Reshape/shapePack.layer_normalization_1/Reshape/shape/0:output:0layer_normalization_1/mul:z:0layer_normalization_1/mul_1:z:0.layer_normalization_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:?
layer_normalization_1/ReshapeReshapeperturbable2/Relu:activations:0,layer_normalization_1/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????@v
!layer_normalization_1/ones/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:e
 layer_normalization_1/ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
layer_normalization_1/onesFill*layer_normalization_1/ones/packed:output:0)layer_normalization_1/ones/Const:output:0*
T0*#
_output_shapes
:?????????w
"layer_normalization_1/zeros/packedPacklayer_normalization_1/mul:z:0*
N*
T0*
_output_shapes
:f
!layer_normalization_1/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
layer_normalization_1/zerosFill+layer_normalization_1/zeros/packed:output:0*layer_normalization_1/zeros/Const:output:0*
T0*#
_output_shapes
:?????????^
layer_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB `
layer_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
&layer_normalization_1/FusedBatchNormV3FusedBatchNormV3&layer_normalization_1/Reshape:output:0#layer_normalization_1/ones:output:0$layer_normalization_1/zeros:output:0$layer_normalization_1/Const:output:0&layer_normalization_1/Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:?
layer_normalization_1/Reshape_1Reshape*layer_normalization_1/FusedBatchNormV3:y:0$layer_normalization_1/Shape:output:0*
T0*'
_output_shapes
:?????????@?
*layer_normalization_1/mul_2/ReadVariableOpReadVariableOp3layer_normalization_1_mul_2_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_1/mul_2Mul(layer_normalization_1/Reshape_1:output:02layer_normalization_1/mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
(layer_normalization_1/add/ReadVariableOpReadVariableOp1layer_normalization_1_add_readvariableop_resource*
_output_shapes
:@*
dtype0?
layer_normalization_1/addAddV2layer_normalization_1/mul_2:z:00layer_normalization_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@?
"perturbable3/MatMul/ReadVariableOpReadVariableOp+perturbable3_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0?
perturbable3/MatMulMatMullayer_normalization_1/add:z:0*perturbable3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
#perturbable3/BiasAdd/ReadVariableOpReadVariableOp,perturbable3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
perturbable3/BiasAddBiasAddperturbable3/MatMul:product:0+perturbable3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????j
perturbable3/TanhTanhperturbable3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????d
IdentityIdentityperturbable3/Tanh:y:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp'^layer_normalization/add/ReadVariableOp)^layer_normalization/mul_2/ReadVariableOp)^layer_normalization_1/add/ReadVariableOp+^layer_normalization_1/mul_2/ReadVariableOp$^perturbable1/BiasAdd/ReadVariableOp#^perturbable1/MatMul/ReadVariableOp$^perturbable2/BiasAdd/ReadVariableOp#^perturbable2/MatMul/ReadVariableOp$^perturbable3/BiasAdd/ReadVariableOp#^perturbable3/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2P
&layer_normalization/add/ReadVariableOp&layer_normalization/add/ReadVariableOp2T
(layer_normalization/mul_2/ReadVariableOp(layer_normalization/mul_2/ReadVariableOp2T
(layer_normalization_1/add/ReadVariableOp(layer_normalization_1/add/ReadVariableOp2X
*layer_normalization_1/mul_2/ReadVariableOp*layer_normalization_1/mul_2/ReadVariableOp2J
#perturbable1/BiasAdd/ReadVariableOp#perturbable1/BiasAdd/ReadVariableOp2H
"perturbable1/MatMul/ReadVariableOp"perturbable1/MatMul/ReadVariableOp2J
#perturbable2/BiasAdd/ReadVariableOp#perturbable2/BiasAdd/ReadVariableOp2H
"perturbable2/MatMul/ReadVariableOp"perturbable2/MatMul/ReadVariableOp2J
#perturbable3/BiasAdd/ReadVariableOp#perturbable3/BiasAdd/ReadVariableOp2H
"perturbable3/MatMul/ReadVariableOp"perturbable3/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_25142114
input_1
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_25141697o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_perturbable1_layer_call_fn_25142375

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?,
?
$__inference__traced_restore_25142621
file_prefix6
$assignvariableop_perturbable1_kernel:@2
$assignvariableop_1_perturbable1_bias:@:
,assignvariableop_2_layer_normalization_gamma:@9
+assignvariableop_3_layer_normalization_beta:@8
&assignvariableop_4_perturbable2_kernel:@@2
$assignvariableop_5_perturbable2_bias:@<
.assignvariableop_6_layer_normalization_1_gamma:@;
-assignvariableop_7_layer_normalization_1_beta:@8
&assignvariableop_8_perturbable3_kernel:@2
$assignvariableop_9_perturbable3_bias:
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp$assignvariableop_perturbable1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_perturbable1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp,assignvariableop_2_layer_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp+assignvariableop_3_layer_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_perturbable2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_perturbable2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp-assignvariableop_7_layer_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_perturbable3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_perturbable3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_11IdentityIdentity_10:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_11Identity_11:output:0*)
_input_shapes
: : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
(__inference_actor_layer_call_fn_25142139

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_25141852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
J__inference_perturbable2_layer_call_and_return_conditional_losses_25142457

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
C__inference_actor_layer_call_and_return_conditional_losses_25141852

inputs'
perturbable1_25141716:@#
perturbable1_25141718:@*
layer_normalization_25141764:@*
layer_normalization_25141766:@'
perturbable2_25141781:@@#
perturbable2_25141783:@,
layer_normalization_1_25141829:@,
layer_normalization_1_25141831:@'
perturbable3_25141846:@#
perturbable3_25141848:
identity??+layer_normalization/StatefulPartitionedCall?-layer_normalization_1/StatefulPartitionedCall?$perturbable1/StatefulPartitionedCall?$perturbable2/StatefulPartitionedCall?$perturbable3/StatefulPartitionedCall?
$perturbable1/StatefulPartitionedCallStatefulPartitionedCallinputsperturbable1_25141716perturbable1_25141718*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable1_layer_call_and_return_conditional_losses_25141715?
+layer_normalization/StatefulPartitionedCallStatefulPartitionedCall-perturbable1/StatefulPartitionedCall:output:0layer_normalization_25141764layer_normalization_25141766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763?
$perturbable2/StatefulPartitionedCallStatefulPartitionedCall4layer_normalization/StatefulPartitionedCall:output:0perturbable2_25141781perturbable2_25141783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable2_layer_call_and_return_conditional_losses_25141780?
-layer_normalization_1/StatefulPartitionedCallStatefulPartitionedCall-perturbable2/StatefulPartitionedCall:output:0layer_normalization_1_25141829layer_normalization_1_25141831*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828?
$perturbable3/StatefulPartitionedCallStatefulPartitionedCall6layer_normalization_1/StatefulPartitionedCall:output:0perturbable3_25141846perturbable3_25141848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_perturbable3_layer_call_and_return_conditional_losses_25141845|
IdentityIdentity-perturbable3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp,^layer_normalization/StatefulPartitionedCall.^layer_normalization_1/StatefulPartitionedCall%^perturbable1/StatefulPartitionedCall%^perturbable2/StatefulPartitionedCall%^perturbable3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 2Z
+layer_normalization/StatefulPartitionedCall+layer_normalization/StatefulPartitionedCall2^
-layer_normalization_1/StatefulPartitionedCall-layer_normalization_1/StatefulPartitionedCall2L
$perturbable1/StatefulPartitionedCall$perturbable1/StatefulPartitionedCall2L
$perturbable2/StatefulPartitionedCall$perturbable2/StatefulPartitionedCall2L
$perturbable3/StatefulPartitionedCall$perturbable3/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25142437

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?!
?
!__inference__traced_save_25142581
file_prefix2
.savev2_perturbable1_kernel_read_readvariableop0
,savev2_perturbable1_bias_read_readvariableop8
4savev2_layer_normalization_gamma_read_readvariableop7
3savev2_layer_normalization_beta_read_readvariableop2
.savev2_perturbable2_kernel_read_readvariableop0
,savev2_perturbable2_bias_read_readvariableop:
6savev2_layer_normalization_1_gamma_read_readvariableop9
5savev2_layer_normalization_1_beta_read_readvariableop2
.savev2_perturbable3_kernel_read_readvariableop0
,savev2_perturbable3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_perturbable1_kernel_read_readvariableop,savev2_perturbable1_bias_read_readvariableop4savev2_layer_normalization_gamma_read_readvariableop3savev2_layer_normalization_beta_read_readvariableop.savev2_perturbable2_kernel_read_readvariableop,savev2_perturbable2_bias_read_readvariableop6savev2_layer_normalization_1_gamma_read_readvariableop5savev2_layer_normalization_1_beta_read_readvariableop.savev2_perturbable3_kernel_read_readvariableop,savev2_perturbable3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*_
_input_shapesN
L: :@:@:@:@:@@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$	 

_output_shapes

:@: 


_output_shapes
::

_output_shapes
: 
?

?
(__inference_actor_layer_call_fn_25142164

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@
	unknown_8:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_actor_layer_call_and_return_conditional_losses_25141981o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
6__inference_layer_normalization_layer_call_fn_25142395

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Z
fURS
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
8__inference_layer_normalization_1_layer_call_fn_25142466

inputs
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25141828o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25141763

inputs+
mul_2_readvariableop_resource:@)
add_readvariableop_resource:@
identity??add/ReadVariableOp?mul_2/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/xConst*
_output_shapes
: *
dtype0*
value	B :S
mulMulmul/x:output:0strided_slice:output:0*
T0*
_output_shapes
: _
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskI
mul_1/xConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulmul_1/x:output:0strided_slice_1:output:0*
T0*
_output_shapes
: Q
Reshape/shape/0Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :?
Reshape/shapePackReshape/shape/0:output:0mul:z:0	mul_1:z:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:l
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????@J
ones/packedPackmul:z:0*
N*
T0*
_output_shapes
:O

ones/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??e
onesFillones/packed:output:0ones/Const:output:0*
T0*#
_output_shapes
:?????????K
zeros/packedPackmul:z:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    h
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*#
_output_shapes
:?????????H
ConstConst*
_output_shapes
: *
dtype0*
valueB J
Const_1Const*
_output_shapes
: *
dtype0*
valueB ?
FusedBatchNormV3FusedBatchNormV3Reshape:output:0ones:output:0zeros:output:0Const:output:0Const_1:output:0*
T0*
U0*o
_output_shapes]
[:?????????@:?????????:?????????:?????????:?????????:*
data_formatNCHW*
epsilon%o?:l
	Reshape_1ReshapeFusedBatchNormV3:y:0Shape:output:0*
T0*'
_output_shapes
:?????????@n
mul_2/ReadVariableOpReadVariableOpmul_2_readvariableop_resource*
_output_shapes
:@*
dtype0p
mul_2MulReshape_1:output:0mul_2/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@j
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:@*
dtype0e
addAddV2	mul_2:z:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:?????????@r
NoOpNoOp^add/ReadVariableOp^mul_2/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 2(
add/ReadVariableOpadd/ReadVariableOp2,
mul_2/ReadVariableOpmul_2/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????@
perturbable30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?b
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
J__call__
*K&call_and_return_all_conditional_losses
L_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	gamma
beta
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
?

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	 gamma
!beta
"	variables
#trainable_variables
$regularization_losses
%	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
?

&kernel
'bias
(	variables
)trainable_variables
*regularization_losses
+	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
f
0
1
2
3
4
5
 6
!7
&8
'9"
trackable_list_wrapper
f
0
1
2
3
4
5
 6
!7
&8
'9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
	regularization_losses
J__call__
L_default_save_signature
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
,
Wserving_default"
signature_map
%:#@2perturbable1/kernel
:@2perturbable1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%@2layer_normalization/gamma
&:$@2layer_normalization/beta
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6non_trainable_variables

7layers
8metrics
9layer_regularization_losses
:layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
%:#@@2perturbable2/kernel
:@2perturbable2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2layer_normalization_1/gamma
(:&@2layer_normalization_1/beta
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
@non_trainable_variables

Alayers
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
"	variables
#trainable_variables
$regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
%:#@2perturbable3/kernel
:2perturbable3/bias
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
(__inference_actor_layer_call_fn_25141875
(__inference_actor_layer_call_fn_25142139
(__inference_actor_layer_call_fn_25142164
(__inference_actor_layer_call_fn_25142029?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_actor_layer_call_and_return_conditional_losses_25142265
C__inference_actor_layer_call_and_return_conditional_losses_25142366
C__inference_actor_layer_call_and_return_conditional_losses_25142058
C__inference_actor_layer_call_and_return_conditional_losses_25142087?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference__wrapped_model_25141697input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_perturbable1_layer_call_fn_25142375?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_perturbable1_layer_call_and_return_conditional_losses_25142386?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
6__inference_layer_normalization_layer_call_fn_25142395?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25142437?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_perturbable2_layer_call_fn_25142446?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_perturbable2_layer_call_and_return_conditional_losses_25142457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
8__inference_layer_normalization_1_layer_call_fn_25142466?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25142508?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_perturbable3_layer_call_fn_25142517?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_perturbable3_layer_call_and_return_conditional_losses_25142528?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_25142114input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_25141697{
 !&'0?-
&?#
!?
input_1?????????
? ";?8
6
perturbable3&?#
perturbable3??????????
C__inference_actor_layer_call_and_return_conditional_losses_25142058m
 !&'8?5
.?+
!?
input_1?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_25142087m
 !&'8?5
.?+
!?
input_1?????????
p

 
? "%?"
?
0?????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_25142265l
 !&'7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
C__inference_actor_layer_call_and_return_conditional_losses_25142366l
 !&'7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
(__inference_actor_layer_call_fn_25141875`
 !&'8?5
.?+
!?
input_1?????????
p 

 
? "???????????
(__inference_actor_layer_call_fn_25142029`
 !&'8?5
.?+
!?
input_1?????????
p

 
? "???????????
(__inference_actor_layer_call_fn_25142139_
 !&'7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
(__inference_actor_layer_call_fn_25142164_
 !&'7?4
-?*
 ?
inputs?????????
p

 
? "???????????
S__inference_layer_normalization_1_layer_call_and_return_conditional_losses_25142508\ !/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
8__inference_layer_normalization_1_layer_call_fn_25142466O !/?,
%?"
 ?
inputs?????????@
? "??????????@?
Q__inference_layer_normalization_layer_call_and_return_conditional_losses_25142437\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
6__inference_layer_normalization_layer_call_fn_25142395O/?,
%?"
 ?
inputs?????????@
? "??????????@?
J__inference_perturbable1_layer_call_and_return_conditional_losses_25142386\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????@
? ?
/__inference_perturbable1_layer_call_fn_25142375O/?,
%?"
 ?
inputs?????????
? "??????????@?
J__inference_perturbable2_layer_call_and_return_conditional_losses_25142457\/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????@
? ?
/__inference_perturbable2_layer_call_fn_25142446O/?,
%?"
 ?
inputs?????????@
? "??????????@?
J__inference_perturbable3_layer_call_and_return_conditional_losses_25142528\&'/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ?
/__inference_perturbable3_layer_call_fn_25142517O&'/?,
%?"
 ?
inputs?????????@
? "???????????
&__inference_signature_wrapper_25142114?
 !&';?8
? 
1?.
,
input_1!?
input_1?????????";?8
6
perturbable3&?#
perturbable3?????????