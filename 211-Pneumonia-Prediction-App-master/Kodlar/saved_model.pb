оп&
д§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02v2.0.0-0-g64c3d382ca8ЁО 
ѓ
conv2d_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*
dtype0*&
_output_shapes
:
r
conv2d_1/biasVarHandleOp*
shared_nameconv2d_1/bias*
dtype0*
_output_shapes
: *
shape:
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
dtype0*
_output_shapes
:
ф
#separable_conv2d_4/depthwise_kernelVarHandleOp*4
shared_name%#separable_conv2d_4/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:
Б
7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/depthwise_kernel*
dtype0*&
_output_shapes
:
ф
#separable_conv2d_4/pointwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *4
shared_name%#separable_conv2d_4/pointwise_kernel
Б
7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_4/pointwise_kernel*
dtype0*&
_output_shapes
: 
є
separable_conv2d_4/biasVarHandleOp*(
shared_nameseparable_conv2d_4/bias*
dtype0*
_output_shapes
: *
shape: 

+separable_conv2d_4/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_4/bias*
dtype0*
_output_shapes
: 
ј
batch_normalization_4/gammaVarHandleOp*
shape: *,
shared_namebatch_normalization_4/gamma*
dtype0*
_output_shapes
: 
Є
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
dtype0*
_output_shapes
: 
ї
batch_normalization_4/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape: *+
shared_namebatch_normalization_4/beta
Ё
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
: *
dtype0
џ
!batch_normalization_4/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape: *2
shared_name#!batch_normalization_4/moving_mean
Њ
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
dtype0*
_output_shapes
: 
б
%batch_normalization_4/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape: *6
shared_name'%batch_normalization_4/moving_variance
Џ
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
dtype0*
_output_shapes
: 
ф
#separable_conv2d_5/depthwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape: *4
shared_name%#separable_conv2d_5/depthwise_kernel
Б
7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/depthwise_kernel*&
_output_shapes
: *
dtype0
ф
#separable_conv2d_5/pointwise_kernelVarHandleOp*
_output_shapes
: *
shape: @*4
shared_name%#separable_conv2d_5/pointwise_kernel*
dtype0
Б
7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_5/pointwise_kernel*
dtype0*&
_output_shapes
: @
є
separable_conv2d_5/biasVarHandleOp*
_output_shapes
: *
shape:@*(
shared_nameseparable_conv2d_5/bias*
dtype0

+separable_conv2d_5/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_5/bias*
dtype0*
_output_shapes
:@
ј
batch_normalization_5/gammaVarHandleOp*
shape:@*,
shared_namebatch_normalization_5/gamma*
dtype0*
_output_shapes
: 
Є
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
dtype0*
_output_shapes
:@
ї
batch_normalization_5/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*+
shared_namebatch_normalization_5/beta
Ё
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
dtype0*
_output_shapes
:@
џ
!batch_normalization_5/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*2
shared_name#!batch_normalization_5/moving_mean
Њ
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
dtype0*
_output_shapes
:@
б
%batch_normalization_5/moving_varianceVarHandleOp*6
shared_name'%batch_normalization_5/moving_variance*
dtype0*
_output_shapes
: *
shape:@
Џ
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
dtype0*
_output_shapes
:@
ф
#separable_conv2d_6/depthwise_kernelVarHandleOp*4
shared_name%#separable_conv2d_6/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:@
Б
7separable_conv2d_6/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/depthwise_kernel*
dtype0*&
_output_shapes
:@
Ф
#separable_conv2d_6/pointwise_kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@ђ*4
shared_name%#separable_conv2d_6/pointwise_kernel
ц
7separable_conv2d_6/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_6/pointwise_kernel*
dtype0*'
_output_shapes
:@ђ
Є
separable_conv2d_6/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*(
shared_nameseparable_conv2d_6/bias
ђ
+separable_conv2d_6/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_6/bias*
dtype0*
_output_shapes	
:ђ
Ј
batch_normalization_6/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*,
shared_namebatch_normalization_6/gamma
ѕ
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
dtype0*
_output_shapes	
:ђ
Ї
batch_normalization_6/betaVarHandleOp*+
shared_namebatch_normalization_6/beta*
dtype0*
_output_shapes
: *
shape:ђ
є
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
dtype0*
_output_shapes	
:ђ
Џ
!batch_normalization_6/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*2
shared_name#!batch_normalization_6/moving_mean
ћ
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:ђ*
dtype0
Б
%batch_normalization_6/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*6
shared_name'%batch_normalization_6/moving_variance
ю
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:ђ*
dtype0
Ф
#separable_conv2d_7/depthwise_kernelVarHandleOp*4
shared_name%#separable_conv2d_7/depthwise_kernel*
dtype0*
_output_shapes
: *
shape:ђ
ц
7separable_conv2d_7/depthwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/depthwise_kernel*
dtype0*'
_output_shapes
:ђ
г
#separable_conv2d_7/pointwise_kernelVarHandleOp*
_output_shapes
: *
shape:ђђ*4
shared_name%#separable_conv2d_7/pointwise_kernel*
dtype0
Ц
7separable_conv2d_7/pointwise_kernel/Read/ReadVariableOpReadVariableOp#separable_conv2d_7/pointwise_kernel*
dtype0*(
_output_shapes
:ђђ
Є
separable_conv2d_7/biasVarHandleOp*
shape:ђ*(
shared_nameseparable_conv2d_7/bias*
dtype0*
_output_shapes
: 
ђ
+separable_conv2d_7/bias/Read/ReadVariableOpReadVariableOpseparable_conv2d_7/bias*
dtype0*
_output_shapes	
:ђ
Ј
batch_normalization_7/gammaVarHandleOp*,
shared_namebatch_normalization_7/gamma*
dtype0*
_output_shapes
: *
shape:ђ
ѕ
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
dtype0*
_output_shapes	
:ђ
Ї
batch_normalization_7/betaVarHandleOp*
shape:ђ*+
shared_namebatch_normalization_7/beta*
dtype0*
_output_shapes
: 
є
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:ђ*
dtype0
Џ
!batch_normalization_7/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*2
shared_name#!batch_normalization_7/moving_mean
ћ
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
dtype0*
_output_shapes	
:ђ
Б
%batch_normalization_7/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*6
shared_name'%batch_normalization_7/moving_variance
ю
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:ђ*
dtype0
z
dense_4/kernelVarHandleOp*
shape:
ђђ*
shared_namedense_4/kernel*
dtype0*
_output_shapes
: 
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
ђђ*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
shape:ђ*
shared_namedense_4/bias*
dtype0
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
dtype0*
_output_shapes	
:ђ
z
dense_5/kernelVarHandleOp*
shape:
ђђ*
shared_namedense_5/kernel*
dtype0*
_output_shapes
: 
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
dtype0* 
_output_shapes
:
ђђ
q
dense_5/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
dtype0*
_output_shapes	
:ђ
y
dense_6/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	ђ@*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
dtype0*
_output_shapes
:	ђ@
p
dense_6/biasVarHandleOp*
_output_shapes
: *
shape:@*
shared_namedense_6/bias*
dtype0
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
x
dense_7/kernelVarHandleOp*
shape
:@*
shared_namedense_7/kernel*
dtype0*
_output_shapes
: 
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
dtype0*
_output_shapes

:@
p
dense_7/biasVarHandleOp*
shared_namedense_7/bias*
dtype0*
_output_shapes
: *
shape:
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
dtype0	*
_output_shapes
: *
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
dtype0	*
_output_shapes
: 
n
RMSprop/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
dtype0*
_output_shapes
: 
~
RMSprop/learning_rateVarHandleOp*
shape: *&
shared_nameRMSprop/learning_rate*
dtype0*
_output_shapes
: 
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
dtype0*
_output_shapes
: 
t
RMSprop/momentumVarHandleOp*
dtype0*
_output_shapes
: *
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
dtype0*
_output_shapes
: 
j
RMSprop/rhoVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
dtype0*
_output_shapes
: 
џ
RMSprop/conv2d_1/kernel/rmsVarHandleOp*,
shared_nameRMSprop/conv2d_1/kernel/rms*
dtype0*
_output_shapes
: *
shape:
Њ
/RMSprop/conv2d_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_1/kernel/rms*
dtype0*&
_output_shapes
:
і
RMSprop/conv2d_1/bias/rmsVarHandleOp**
shared_nameRMSprop/conv2d_1/bias/rms*
dtype0*
_output_shapes
: *
shape:
Ѓ
-RMSprop/conv2d_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_1/bias/rms*
dtype0*
_output_shapes
:
┬
/RMSprop/separable_conv2d_4/depthwise_kernel/rmsVarHandleOp*
shape:*@
shared_name1/RMSprop/separable_conv2d_4/depthwise_kernel/rms*
dtype0*
_output_shapes
: 
╗
CRMSprop/separable_conv2d_4/depthwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_4/depthwise_kernel/rms*
dtype0*&
_output_shapes
:
┬
/RMSprop/separable_conv2d_4/pointwise_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: *@
shared_name1/RMSprop/separable_conv2d_4/pointwise_kernel/rms
╗
CRMSprop/separable_conv2d_4/pointwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_4/pointwise_kernel/rms*
dtype0*&
_output_shapes
: 
ъ
#RMSprop/separable_conv2d_4/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: *4
shared_name%#RMSprop/separable_conv2d_4/bias/rms
Ќ
7RMSprop/separable_conv2d_4/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/separable_conv2d_4/bias/rms*
dtype0*
_output_shapes
: 
д
'RMSprop/batch_normalization_4/gamma/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: *8
shared_name)'RMSprop/batch_normalization_4/gamma/rms
Ъ
;RMSprop/batch_normalization_4/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_4/gamma/rms*
dtype0*
_output_shapes
: 
ц
&RMSprop/batch_normalization_4/beta/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: *7
shared_name(&RMSprop/batch_normalization_4/beta/rms
Ю
:RMSprop/batch_normalization_4/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_4/beta/rms*
dtype0*
_output_shapes
: 
┬
/RMSprop/separable_conv2d_5/depthwise_kernel/rmsVarHandleOp*
_output_shapes
: *
shape: *@
shared_name1/RMSprop/separable_conv2d_5/depthwise_kernel/rms*
dtype0
╗
CRMSprop/separable_conv2d_5/depthwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_5/depthwise_kernel/rms*
dtype0*&
_output_shapes
: 
┬
/RMSprop/separable_conv2d_5/pointwise_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape: @*@
shared_name1/RMSprop/separable_conv2d_5/pointwise_kernel/rms
╗
CRMSprop/separable_conv2d_5/pointwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_5/pointwise_kernel/rms*
dtype0*&
_output_shapes
: @
ъ
#RMSprop/separable_conv2d_5/bias/rmsVarHandleOp*
shape:@*4
shared_name%#RMSprop/separable_conv2d_5/bias/rms*
dtype0*
_output_shapes
: 
Ќ
7RMSprop/separable_conv2d_5/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/separable_conv2d_5/bias/rms*
dtype0*
_output_shapes
:@
д
'RMSprop/batch_normalization_5/gamma/rmsVarHandleOp*8
shared_name)'RMSprop/batch_normalization_5/gamma/rms*
dtype0*
_output_shapes
: *
shape:@
Ъ
;RMSprop/batch_normalization_5/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_5/gamma/rms*
dtype0*
_output_shapes
:@
ц
&RMSprop/batch_normalization_5/beta/rmsVarHandleOp*
shape:@*7
shared_name(&RMSprop/batch_normalization_5/beta/rms*
dtype0*
_output_shapes
: 
Ю
:RMSprop/batch_normalization_5/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_5/beta/rms*
_output_shapes
:@*
dtype0
┬
/RMSprop/separable_conv2d_6/depthwise_kernel/rmsVarHandleOp*@
shared_name1/RMSprop/separable_conv2d_6/depthwise_kernel/rms*
dtype0*
_output_shapes
: *
shape:@
╗
CRMSprop/separable_conv2d_6/depthwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_6/depthwise_kernel/rms*
dtype0*&
_output_shapes
:@
├
/RMSprop/separable_conv2d_6/pointwise_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:@ђ*@
shared_name1/RMSprop/separable_conv2d_6/pointwise_kernel/rms
╝
CRMSprop/separable_conv2d_6/pointwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_6/pointwise_kernel/rms*
dtype0*'
_output_shapes
:@ђ
Ъ
#RMSprop/separable_conv2d_6/bias/rmsVarHandleOp*
shape:ђ*4
shared_name%#RMSprop/separable_conv2d_6/bias/rms*
dtype0*
_output_shapes
: 
ў
7RMSprop/separable_conv2d_6/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/separable_conv2d_6/bias/rms*
dtype0*
_output_shapes	
:ђ
Д
'RMSprop/batch_normalization_6/gamma/rmsVarHandleOp*8
shared_name)'RMSprop/batch_normalization_6/gamma/rms*
dtype0*
_output_shapes
: *
shape:ђ
а
;RMSprop/batch_normalization_6/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_6/gamma/rms*
dtype0*
_output_shapes	
:ђ
Ц
&RMSprop/batch_normalization_6/beta/rmsVarHandleOp*
shape:ђ*7
shared_name(&RMSprop/batch_normalization_6/beta/rms*
dtype0*
_output_shapes
: 
ъ
:RMSprop/batch_normalization_6/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_6/beta/rms*
dtype0*
_output_shapes	
:ђ
├
/RMSprop/separable_conv2d_7/depthwise_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*@
shared_name1/RMSprop/separable_conv2d_7/depthwise_kernel/rms
╝
CRMSprop/separable_conv2d_7/depthwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_7/depthwise_kernel/rms*
dtype0*'
_output_shapes
:ђ
─
/RMSprop/separable_conv2d_7/pointwise_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђђ*@
shared_name1/RMSprop/separable_conv2d_7/pointwise_kernel/rms
й
CRMSprop/separable_conv2d_7/pointwise_kernel/rms/Read/ReadVariableOpReadVariableOp/RMSprop/separable_conv2d_7/pointwise_kernel/rms*
dtype0*(
_output_shapes
:ђђ
Ъ
#RMSprop/separable_conv2d_7/bias/rmsVarHandleOp*
shape:ђ*4
shared_name%#RMSprop/separable_conv2d_7/bias/rms*
dtype0*
_output_shapes
: 
ў
7RMSprop/separable_conv2d_7/bias/rms/Read/ReadVariableOpReadVariableOp#RMSprop/separable_conv2d_7/bias/rms*
dtype0*
_output_shapes	
:ђ
Д
'RMSprop/batch_normalization_7/gamma/rmsVarHandleOp*8
shared_name)'RMSprop/batch_normalization_7/gamma/rms*
dtype0*
_output_shapes
: *
shape:ђ
а
;RMSprop/batch_normalization_7/gamma/rms/Read/ReadVariableOpReadVariableOp'RMSprop/batch_normalization_7/gamma/rms*
dtype0*
_output_shapes	
:ђ
Ц
&RMSprop/batch_normalization_7/beta/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:ђ*7
shared_name(&RMSprop/batch_normalization_7/beta/rms
ъ
:RMSprop/batch_normalization_7/beta/rms/Read/ReadVariableOpReadVariableOp&RMSprop/batch_normalization_7/beta/rms*
dtype0*
_output_shapes	
:ђ
њ
RMSprop/dense_4/kernel/rmsVarHandleOp*+
shared_nameRMSprop/dense_4/kernel/rms*
dtype0*
_output_shapes
: *
shape:
ђђ
І
.RMSprop/dense_4/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/kernel/rms*
dtype0* 
_output_shapes
:
ђђ
Ѕ
RMSprop/dense_4/bias/rmsVarHandleOp*)
shared_nameRMSprop/dense_4/bias/rms*
dtype0*
_output_shapes
: *
shape:ђ
ѓ
,RMSprop/dense_4/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_4/bias/rms*
dtype0*
_output_shapes	
:ђ
њ
RMSprop/dense_5/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:
ђђ*+
shared_nameRMSprop/dense_5/kernel/rms
І
.RMSprop/dense_5/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_5/kernel/rms*
dtype0* 
_output_shapes
:
ђђ
Ѕ
RMSprop/dense_5/bias/rmsVarHandleOp*
_output_shapes
: *
shape:ђ*)
shared_nameRMSprop/dense_5/bias/rms*
dtype0
ѓ
,RMSprop/dense_5/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_5/bias/rms*
_output_shapes	
:ђ*
dtype0
Љ
RMSprop/dense_6/kernel/rmsVarHandleOp*+
shared_nameRMSprop/dense_6/kernel/rms*
dtype0*
_output_shapes
: *
shape:	ђ@
і
.RMSprop/dense_6/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_6/kernel/rms*
dtype0*
_output_shapes
:	ђ@
ѕ
RMSprop/dense_6/bias/rmsVarHandleOp*)
shared_nameRMSprop/dense_6/bias/rms*
dtype0*
_output_shapes
: *
shape:@
Ђ
,RMSprop/dense_6/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_6/bias/rms*
dtype0*
_output_shapes
:@
љ
RMSprop/dense_7/kernel/rmsVarHandleOp*+
shared_nameRMSprop/dense_7/kernel/rms*
dtype0*
_output_shapes
: *
shape
:@
Ѕ
.RMSprop/dense_7/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/kernel/rms*
dtype0*
_output_shapes

:@
ѕ
RMSprop/dense_7/bias/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:*)
shared_nameRMSprop/dense_7/bias/rms
Ђ
,RMSprop/dense_7/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_7/bias/rms*
dtype0*
_output_shapes
:

NoOpNoOp
▀Ќ
ConstConst"/device:CPU:0*ЎЌ
valueјЌBіЌ BѓЌ
џ
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
R
 	variables
!trainable_variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
R
*	variables
+trainable_variables
,regularization_losses
-	keras_api
ѕ
.depthwise_kernel
/pointwise_kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
Ќ
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
R
>	variables
?trainable_variables
@regularization_losses
A	keras_api
ѕ
Bdepthwise_kernel
Cpointwise_kernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
Ќ
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
ѕ
Vdepthwise_kernel
Wpointwise_kernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
Ќ
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
R
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
R
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
ѕ
ndepthwise_kernel
opointwise_kernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
Ќ
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
T
~	variables
trainable_variables
ђregularization_losses
Ђ	keras_api
V
ѓ	variables
Ѓtrainable_variables
ёregularization_losses
Ё	keras_api
V
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
n
іkernel
	Іbias
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
V
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
n
ћkernel
	Ћbias
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
V
џ	variables
Џtrainable_variables
юregularization_losses
Ю	keras_api
n
ъkernel
	Ъbias
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
V
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
n
еkernel
	Еbias
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
┤
	«iter

»decay
░learning_rate
▒momentum
▓rho
$rmsД
%rmsе
.rmsЕ
/rmsф
0rmsФ
6rmsг
7rmsГ
Brms«
Crms»
Drms░
Jrms▒
Krms▓
Vrms│
Wrms┤
Xrmsх
^rmsХ
_rmsи
nrmsИ
orms╣
prms║
vrms╗
wrms╝іrmsйІrmsЙћrms┐Ћrms└ъrms┴Ъrms┬еrms├Еrms─
«
$0
%1
.2
/3
04
65
76
87
98
B9
C10
D11
J12
K13
L14
M15
V16
W17
X18
^19
_20
`21
a22
n23
o24
p25
v26
w27
x28
y29
і30
І31
ћ32
Ћ33
ъ34
Ъ35
е36
Е37
Ь
$0
%1
.2
/3
04
65
76
B7
C8
D9
J10
K11
V12
W13
X14
^15
_16
n17
o18
p19
v20
w21
і22
І23
ћ24
Ћ25
ъ26
Ъ27
е28
Е29
 
ъ
│layers
	variables
trainable_variables
regularization_losses
 ┤layer_regularization_losses
хmetrics
Хnon_trainable_variables
 
 
 
 
ъ
иlayers
 	variables
!trainable_variables
"regularization_losses
 Иlayer_regularization_losses
╣metrics
║non_trainable_variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
ъ
╗layers
&	variables
'trainable_variables
(regularization_losses
 ╝layer_regularization_losses
йmetrics
Йnon_trainable_variables
 
 
 
ъ
┐layers
*	variables
+trainable_variables
,regularization_losses
 └layer_regularization_losses
┴metrics
┬non_trainable_variables
yw
VARIABLE_VALUE#separable_conv2d_4/depthwise_kernel@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_4/pointwise_kernel@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1
02

.0
/1
02
 
ъ
├layers
1	variables
2trainable_variables
3regularization_losses
 ─layer_regularization_losses
┼metrics
кnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

60
71
82
93

60
71
 
ъ
Кlayers
:	variables
;trainable_variables
<regularization_losses
 ╚layer_regularization_losses
╔metrics
╩non_trainable_variables
 
 
 
ъ
╦layers
>	variables
?trainable_variables
@regularization_losses
 ╠layer_regularization_losses
═metrics
╬non_trainable_variables
yw
VARIABLE_VALUE#separable_conv2d_5/depthwise_kernel@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_5/pointwise_kernel@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_5/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
D2

B0
C1
D2
 
ъ
¤layers
E	variables
Ftrainable_variables
Gregularization_losses
 лlayer_regularization_losses
Лmetrics
мnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_5/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_5/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_5/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_5/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
L2
M3

J0
K1
 
ъ
Мlayers
N	variables
Otrainable_variables
Pregularization_losses
 нlayer_regularization_losses
Нmetrics
оnon_trainable_variables
 
 
 
ъ
Оlayers
R	variables
Strainable_variables
Tregularization_losses
 пlayer_regularization_losses
┘metrics
┌non_trainable_variables
yw
VARIABLE_VALUE#separable_conv2d_6/depthwise_kernel@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_6/pointwise_kernel@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1
X2

V0
W1
X2
 
ъ
█layers
Y	variables
Ztrainable_variables
[regularization_losses
 ▄layer_regularization_losses
Пmetrics
яnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_6/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_6/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_6/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_6/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
`2
a3

^0
_1
 
ъ
▀layers
b	variables
ctrainable_variables
dregularization_losses
 Яlayer_regularization_losses
рmetrics
Рnon_trainable_variables
 
 
 
ъ
сlayers
f	variables
gtrainable_variables
hregularization_losses
 Сlayer_regularization_losses
тmetrics
Тnon_trainable_variables
 
 
 
ъ
уlayers
j	variables
ktrainable_variables
lregularization_losses
 Уlayer_regularization_losses
жmetrics
Жnon_trainable_variables
yw
VARIABLE_VALUE#separable_conv2d_7/depthwise_kernel@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE#separable_conv2d_7/pointwise_kernel@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEseparable_conv2d_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

n0
o1
p2

n0
o1
p2
 
ъ
вlayers
q	variables
rtrainable_variables
sregularization_losses
 Вlayer_regularization_losses
ьmetrics
Ьnon_trainable_variables
 
fd
VARIABLE_VALUEbatch_normalization_7/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_7/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_7/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_7/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
x2
y3

v0
w1
 
ъ
№layers
z	variables
{trainable_variables
|regularization_losses
 ­layer_regularization_losses
ыmetrics
Ыnon_trainable_variables
 
 
 
Ъ
зlayers
~	variables
trainable_variables
ђregularization_losses
 Зlayer_regularization_losses
шmetrics
Шnon_trainable_variables
 
 
 
А
эlayers
ѓ	variables
Ѓtrainable_variables
ёregularization_losses
 Эlayer_regularization_losses
щmetrics
Щnon_trainable_variables
 
 
 
А
чlayers
є	variables
Єtrainable_variables
ѕregularization_losses
 Чlayer_regularization_losses
§metrics
■non_trainable_variables
ZX
VARIABLE_VALUEdense_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_4/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

і0
І1

і0
І1
 
А
 layers
ї	variables
Їtrainable_variables
јregularization_losses
 ђlayer_regularization_losses
Ђmetrics
ѓnon_trainable_variables
 
 
 
А
Ѓlayers
љ	variables
Љtrainable_variables
њregularization_losses
 ёlayer_regularization_losses
Ёmetrics
єnon_trainable_variables
[Y
VARIABLE_VALUEdense_5/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_5/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

ћ0
Ћ1

ћ0
Ћ1
 
А
Єlayers
ќ	variables
Ќtrainable_variables
ўregularization_losses
 ѕlayer_regularization_losses
Ѕmetrics
іnon_trainable_variables
 
 
 
А
Іlayers
џ	variables
Џtrainable_variables
юregularization_losses
 їlayer_regularization_losses
Їmetrics
јnon_trainable_variables
[Y
VARIABLE_VALUEdense_6/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_6/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

ъ0
Ъ1

ъ0
Ъ1
 
А
Јlayers
а	variables
Аtrainable_variables
бregularization_losses
 љlayer_regularization_losses
Љmetrics
њnon_trainable_variables
 
 
 
А
Њlayers
ц	variables
Цtrainable_variables
дregularization_losses
 ћlayer_regularization_losses
Ћmetrics
ќnon_trainable_variables
[Y
VARIABLE_VALUEdense_7/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_7/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

е0
Е1

е0
Е1
 
А
Ќlayers
ф	variables
Фtrainable_variables
гregularization_losses
 ўlayer_regularization_losses
Ўmetrics
џnon_trainable_variables
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
Х
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
 

Џ0
8
80
91
L2
M3
`4
a5
x6
y7
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

80
91
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

L0
M1
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

`0
a1
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

x0
y1
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


юtotal

Юcount
ъ
_fn_kwargs
Ъ	variables
аtrainable_variables
Аregularization_losses
б	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 

ю0
Ю1
 
 
А
Бlayers
Ъ	variables
аtrainable_variables
Аregularization_losses
 цlayer_regularization_losses
Цmetrics
дnon_trainable_variables
 
 
 

ю0
Ю1
єЃ
VARIABLE_VALUERMSprop/conv2d_1/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUERMSprop/conv2d_1/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_4/depthwise_kernel/rms^layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_4/pointwise_kernel/rms^layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#RMSprop/separable_conv2d_4/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE'RMSprop/batch_normalization_4/gamma/rmsSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE&RMSprop/batch_normalization_4/beta/rmsRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_5/depthwise_kernel/rms^layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_5/pointwise_kernel/rms^layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#RMSprop/separable_conv2d_5/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE'RMSprop/batch_normalization_5/gamma/rmsSlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE&RMSprop/batch_normalization_5/beta/rmsRlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_6/depthwise_kernel/rms^layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_6/pointwise_kernel/rms^layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#RMSprop/separable_conv2d_6/bias/rmsRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE'RMSprop/batch_normalization_6/gamma/rmsSlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE&RMSprop/batch_normalization_6/beta/rmsRlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_7/depthwise_kernel/rms^layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
цА
VARIABLE_VALUE/RMSprop/separable_conv2d_7/pointwise_kernel/rms^layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#RMSprop/separable_conv2d_7/bias/rmsRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Љј
VARIABLE_VALUE'RMSprop/batch_normalization_7/gamma/rmsSlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Јї
VARIABLE_VALUE&RMSprop/batch_normalization_7/beta/rmsRlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ёѓ
VARIABLE_VALUERMSprop/dense_4/kernel/rmsTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
ђ~
VARIABLE_VALUERMSprop/dense_4/bias/rmsRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUERMSprop/dense_5/kernel/rmsUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUERMSprop/dense_5/bias/rmsSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUERMSprop/dense_6/kernel/rmsUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUERMSprop/dense_6/bias/rmsSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
єЃ
VARIABLE_VALUERMSprop/dense_7/kernel/rmsUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
Ђ
VARIABLE_VALUERMSprop/dense_7/bias/rmsSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Љ
serving_default_conv2d_1_inputPlaceholder*
dtype0*/
_output_shapes
:         dd*$
shape:         dd
§

StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_1_inputconv2d_1/kernelconv2d_1/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelseparable_conv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelseparable_conv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*-
config_proto

GPU

CPU2*0J 8*2
Tin+
)2'*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-2659205*.
f)R'
%__inference_signature_wrapper_2657568*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
│ 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp7separable_conv2d_4/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_4/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_4/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp7separable_conv2d_5/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_5/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_5/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp7separable_conv2d_6/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_6/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_6/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp7separable_conv2d_7/depthwise_kernel/Read/ReadVariableOp7separable_conv2d_7/pointwise_kernel/Read/ReadVariableOp+separable_conv2d_7/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp/RMSprop/conv2d_1/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_1/bias/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_4/depthwise_kernel/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_4/pointwise_kernel/rms/Read/ReadVariableOp7RMSprop/separable_conv2d_4/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_4/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_4/beta/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_5/depthwise_kernel/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_5/pointwise_kernel/rms/Read/ReadVariableOp7RMSprop/separable_conv2d_5/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_5/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_5/beta/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_6/depthwise_kernel/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_6/pointwise_kernel/rms/Read/ReadVariableOp7RMSprop/separable_conv2d_6/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_6/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_6/beta/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_7/depthwise_kernel/rms/Read/ReadVariableOpCRMSprop/separable_conv2d_7/pointwise_kernel/rms/Read/ReadVariableOp7RMSprop/separable_conv2d_7/bias/rms/Read/ReadVariableOp;RMSprop/batch_normalization_7/gamma/rms/Read/ReadVariableOp:RMSprop/batch_normalization_7/beta/rms/Read/ReadVariableOp.RMSprop/dense_4/kernel/rms/Read/ReadVariableOp,RMSprop/dense_4/bias/rms/Read/ReadVariableOp.RMSprop/dense_5/kernel/rms/Read/ReadVariableOp,RMSprop/dense_5/bias/rms/Read/ReadVariableOp.RMSprop/dense_6/kernel/rms/Read/ReadVariableOp,RMSprop/dense_6/bias/rms/Read/ReadVariableOp.RMSprop/dense_7/kernel/rms/Read/ReadVariableOp,RMSprop/dense_7/bias/rms/Read/ReadVariableOpConst*-
config_proto

GPU

CPU2*0J 8*
_output_shapes
: *X
TinQ
O2M	*.
_gradient_op_typePartitionedCall-2659302*)
f$R"
 __inference__traced_save_2659301*
Tout
2
м
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/bias#separable_conv2d_4/depthwise_kernel#separable_conv2d_4/pointwise_kernelseparable_conv2d_4/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#separable_conv2d_5/depthwise_kernel#separable_conv2d_5/pointwise_kernelseparable_conv2d_5/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_variance#separable_conv2d_6/depthwise_kernel#separable_conv2d_6/pointwise_kernelseparable_conv2d_6/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#separable_conv2d_7/depthwise_kernel#separable_conv2d_7/pointwise_kernelseparable_conv2d_7/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancedense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcountRMSprop/conv2d_1/kernel/rmsRMSprop/conv2d_1/bias/rms/RMSprop/separable_conv2d_4/depthwise_kernel/rms/RMSprop/separable_conv2d_4/pointwise_kernel/rms#RMSprop/separable_conv2d_4/bias/rms'RMSprop/batch_normalization_4/gamma/rms&RMSprop/batch_normalization_4/beta/rms/RMSprop/separable_conv2d_5/depthwise_kernel/rms/RMSprop/separable_conv2d_5/pointwise_kernel/rms#RMSprop/separable_conv2d_5/bias/rms'RMSprop/batch_normalization_5/gamma/rms&RMSprop/batch_normalization_5/beta/rms/RMSprop/separable_conv2d_6/depthwise_kernel/rms/RMSprop/separable_conv2d_6/pointwise_kernel/rms#RMSprop/separable_conv2d_6/bias/rms'RMSprop/batch_normalization_6/gamma/rms&RMSprop/batch_normalization_6/beta/rms/RMSprop/separable_conv2d_7/depthwise_kernel/rms/RMSprop/separable_conv2d_7/pointwise_kernel/rms#RMSprop/separable_conv2d_7/bias/rms'RMSprop/batch_normalization_7/gamma/rms&RMSprop/batch_normalization_7/beta/rmsRMSprop/dense_4/kernel/rmsRMSprop/dense_4/bias/rmsRMSprop/dense_5/kernel/rmsRMSprop/dense_5/bias/rmsRMSprop/dense_6/kernel/rmsRMSprop/dense_6/bias/rmsRMSprop/dense_7/kernel/rmsRMSprop/dense_7/bias/rms*-
config_proto

GPU

CPU2*0J 8*W
TinP
N2L*
_output_shapes
: *.
_gradient_op_typePartitionedCall-2659540*,
f'R%
#__inference__traced_restore_2659539*
Tout
2╚¤
│
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657043

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
dtype0*(
_output_shapes
:         ђ*
T0ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:         ђ*
T0Ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         ђb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         ђ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:         ђ*
T0Z
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
ѕy
§
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657236
conv2d_1_input+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_45
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_45
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_45
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб*separable_conv2d_4/StatefulPartitionedCallб*separable_conv2d_5/StatefulPartitionedCallб*separable_conv2d_6/StatefulPartitionedCallб*separable_conv2d_7/StatefulPartitionedCallю
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_input'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         dd*
Tin
2*.
_gradient_op_typePartitionedCall-2655672*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666*
Tout
2р
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2655691*U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         22њ
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         22 *
Tin
2*.
_gradient_op_typePartitionedCall-2655719*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713*
Tout
2с
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656525*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         22 *
Tin	
2*.
_gradient_op_typePartitionedCall-2656538Ь
max_pooling2d_6/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:          *.
_gradient_op_typePartitionedCall-2655881*U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875*
Tout
2*-
config_proto

GPU

CPU2*0J 8њ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2655909*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903*
Tout
2с
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656626*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656613*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         @Ь
max_pooling2d_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         @*
Tin
2*.
_gradient_op_typePartitionedCall-2656071*U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065*
Tout
2Њ
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*X
fSRQ
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656099С
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656714*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656701№
max_pooling2d_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656261*U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255*
Tout
2Н
dropout_5/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656767*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656755*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2Ї
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*X
fSRQ
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656289С
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656846*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656833*
Tout
2№
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656451*U
fPRN
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445*
Tout
2*-
config_proto

GPU

CPU2*0J 8Н
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656899*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656887*
Tout
2К
flatten_1/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656916*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910*
Tout
2Ц
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2656940*M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2═
dropout_7/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656990*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656978Ц
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657012*M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ═
dropout_8/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2657062*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657050*
Tout
2ц
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657084*M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         @╠
dropout_9/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2657134*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657122*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         @ц
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657156*M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tin
2Ј
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall: : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :. *
(
_user_specified_nameconv2d_1_input: : : : : : : : :	 :
 : : : 
┘
ш
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658367

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         @:@:@:@:@:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:         @*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: :& "
 
_user_specified_nameinputs: : : 
Ј
ш
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658277

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+                            : : : : :*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: : : : :& "
 
_user_specified_nameinputs
Т/
Ќ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658255

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*]
_output_shapesK
I:+                            : : : : :*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: └
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: ─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpЩ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                            *
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
■
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2658869

inputs
identity^
Reshape/shapeConst*
valueB"     	  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:         ђ*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
│
M
1__inference_max_pooling2d_8_layer_call_fn_2656264

inputs
identity╔
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*J
_output_shapes8
6:4                                    *.
_gradient_op_typePartitionedCall-2656261*U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255*
Tout
2Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
«
ђ
7__inference_batch_normalization_4_layer_call_fn_2658210

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         22 *.
_gradient_op_typePartitionedCall-2656528*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656503*
Tout
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         22 "
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
┴
d
+__inference_dropout_9_layer_call_fn_2659028

inputs
identityѕбStatefulPartitionedCall░
StatefulPartitionedCallStatefulPartitionedCallinputs*'
_output_shapes
:         @*
Tin
2*.
_gradient_op_typePartitionedCall-2657126*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657115*
Tout
2*-
config_proto

GPU

CPU2*0J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*&
_input_shapes
:         @22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
џ
ш
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656241

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0и
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: р
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
ф
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659018

inputs
identityѕQ
dropout/rateConst*
valueB
 *џЎ>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         @ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         @ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         @R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         @a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         @i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*&
_input_shapes
:         @:& "
 
_user_specified_nameinputs
«
ђ
7__inference_batch_normalization_5_layer_call_fn_2658385

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2656626*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656613*
Tout
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
▀
ф
)__inference_dense_5_layer_call_fn_2658945

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657012*M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђЃ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
ъ
h
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685

inputs
identityб
MaxPoolMaxPoolinputs*
ksize
*
paddingVALID*J
_output_shapes8
6:4                                    *
strides
{
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
я
л
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбseparable_conv2d/ReadVariableOpб!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:┬
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: o
separable_conv2d/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0o
separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:п
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*A
_output_shapes/
-:+                           *
T0*
strides
*
paddingSAME▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
paddingVALID*A
_output_shapes/
-:+                            *
T0*
strides
а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ў
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                            j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                            М
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*A
_output_shapes/
-:+                            *
T0"
identityIdentity:output:0*L
_input_shapes;
9:+                           :::2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : 
│
M
1__inference_max_pooling2d_5_layer_call_fn_2655694

inputs
identity╔
PartitionedCallPartitionedCallinputs*U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*J
_output_shapes8
6:4                                    *.
_gradient_op_typePartitionedCall-2655691Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
п	
П
D__inference_dense_5_layer_call_and_return_conditional_losses_2658938

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
│
M
1__inference_max_pooling2d_6_layer_call_fn_2655884

inputs
identity╔
PartitionedCallPartitionedCallinputs*U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*J
_output_shapes8
6:4                                    *.
_gradient_op_typePartitionedCall-2655881Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
С
ђ
7__inference_batch_normalization_5_layer_call_fn_2658452

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656018*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656017*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*A
_output_shapes/
-:+                           @ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
С
л
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбseparable_conv2d/ReadVariableOpб!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@├
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@ђo
separable_conv2d/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:o
separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:п
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+                           @*
T0Я
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
paddingVALID*B
_output_shapes0
.:,                           ђ*
T0*
strides
А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђџ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,                           ђk
ReluReluBiasAdd:output:0*B
_output_shapes0
.:,                           ђ*
T0н
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : 
я
л
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбseparable_conv2d/ReadVariableOpб!separable_conv2d/ReadVariableOp_1Й
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ┬
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @o
separable_conv2d/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:o
separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:п
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+                            *
T0▀
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
T0*
strides
*
paddingVALID*A
_output_shapes/
-:+                           @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ў
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+                           @j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           @М
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_12B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp: :& "
 
_user_specified_nameinputs: : 
Ѕ
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656978

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:         ђ*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
п	
П
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
С
ш
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658734

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ╗
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0¤
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs
С
ђ
7__inference_batch_normalization_4_layer_call_fn_2658286

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*A
_output_shapes/
-:+                            *.
_gradient_op_typePartitionedCall-2655828*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2655827*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
С
ђ
7__inference_batch_normalization_4_layer_call_fn_2658295

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*A
_output_shapes/
-:+                            *
Tin	
2*.
_gradient_op_typePartitionedCall-2655862*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2655861*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                            *
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
Яђ
▒
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657168
conv2d_1_input+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_45
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_45
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_45
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_5/StatefulPartitionedCallб!dropout_6/StatefulPartitionedCallб!dropout_7/StatefulPartitionedCallб!dropout_8/StatefulPartitionedCallб!dropout_9/StatefulPartitionedCallб*separable_conv2d_4/StatefulPartitionedCallб*separable_conv2d_5/StatefulPartitionedCallб*separable_conv2d_6/StatefulPartitionedCallб*separable_conv2d_7/StatefulPartitionedCallю
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallconv2d_1_input'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         dd*.
_gradient_op_typePartitionedCall-2655672*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666*
Tout
2р
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         22*.
_gradient_op_typePartitionedCall-2655691*U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685*
Tout
2њ
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*.
_gradient_op_typePartitionedCall-2655719*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         22 с
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         22 *.
_gradient_op_typePartitionedCall-2656528*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656503*
Tout
2Ь
max_pooling2d_6/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:          *
Tin
2*.
_gradient_op_typePartitionedCall-2655881*U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875*
Tout
2њ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2655909*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903*
Tout
2*-
config_proto

GPU

CPU2*0J 8с
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         @*
Tin	
2*.
_gradient_op_typePartitionedCall-2656616*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656591*
Tout
2Ь
max_pooling2d_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         @*
Tin
2*.
_gradient_op_typePartitionedCall-2656071*U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065*
Tout
2Њ
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656099*X
fSRQ
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093*
Tout
2С
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656704*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656679№
max_pooling2d_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656261*U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђт
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656748*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656759Ћ
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656289*X
fSRQ
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283С
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656836*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656811*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ№
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656451*U
fPRN
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445*
Tout
2Ѕ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656891*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656880*
Tout
2¤
flatten_1/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656916*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђЦ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656940*M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934*
Tout
2Ђ
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656971*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656982Г
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2657012*M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006Ђ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2657054*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657043*
Tout
2г
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         @*
Tin
2*.
_gradient_op_typePartitionedCall-2657084ђ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-2657126*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657115*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         @г
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tin
2*.
_gradient_op_typePartitionedCall-2657156*M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150├
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
п
G
+__inference_dropout_6_layer_call_fn_2658863

inputs
identityЕ
PartitionedCallPartitionedCallinputs*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656899*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656887*
Tout
2*-
config_proto

GPU

CPU2*0J 8i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
┼ц
К-
#__inference__traced_restore_2659539
file_prefix$
 assignvariableop_conv2d_1_kernel$
 assignvariableop_1_conv2d_1_bias:
6assignvariableop_2_separable_conv2d_4_depthwise_kernel:
6assignvariableop_3_separable_conv2d_4_pointwise_kernel.
*assignvariableop_4_separable_conv2d_4_bias2
.assignvariableop_5_batch_normalization_4_gamma1
-assignvariableop_6_batch_normalization_4_beta8
4assignvariableop_7_batch_normalization_4_moving_mean<
8assignvariableop_8_batch_normalization_4_moving_variance:
6assignvariableop_9_separable_conv2d_5_depthwise_kernel;
7assignvariableop_10_separable_conv2d_5_pointwise_kernel/
+assignvariableop_11_separable_conv2d_5_bias3
/assignvariableop_12_batch_normalization_5_gamma2
.assignvariableop_13_batch_normalization_5_beta9
5assignvariableop_14_batch_normalization_5_moving_mean=
9assignvariableop_15_batch_normalization_5_moving_variance;
7assignvariableop_16_separable_conv2d_6_depthwise_kernel;
7assignvariableop_17_separable_conv2d_6_pointwise_kernel/
+assignvariableop_18_separable_conv2d_6_bias3
/assignvariableop_19_batch_normalization_6_gamma2
.assignvariableop_20_batch_normalization_6_beta9
5assignvariableop_21_batch_normalization_6_moving_mean=
9assignvariableop_22_batch_normalization_6_moving_variance;
7assignvariableop_23_separable_conv2d_7_depthwise_kernel;
7assignvariableop_24_separable_conv2d_7_pointwise_kernel/
+assignvariableop_25_separable_conv2d_7_bias3
/assignvariableop_26_batch_normalization_7_gamma2
.assignvariableop_27_batch_normalization_7_beta9
5assignvariableop_28_batch_normalization_7_moving_mean=
9assignvariableop_29_batch_normalization_7_moving_variance&
"assignvariableop_30_dense_4_kernel$
 assignvariableop_31_dense_4_bias&
"assignvariableop_32_dense_5_kernel$
 assignvariableop_33_dense_5_bias&
"assignvariableop_34_dense_6_kernel$
 assignvariableop_35_dense_6_bias&
"assignvariableop_36_dense_7_kernel$
 assignvariableop_37_dense_7_bias$
 assignvariableop_38_rmsprop_iter%
!assignvariableop_39_rmsprop_decay-
)assignvariableop_40_rmsprop_learning_rate(
$assignvariableop_41_rmsprop_momentum#
assignvariableop_42_rmsprop_rho
assignvariableop_43_total
assignvariableop_44_count3
/assignvariableop_45_rmsprop_conv2d_1_kernel_rms1
-assignvariableop_46_rmsprop_conv2d_1_bias_rmsG
Cassignvariableop_47_rmsprop_separable_conv2d_4_depthwise_kernel_rmsG
Cassignvariableop_48_rmsprop_separable_conv2d_4_pointwise_kernel_rms;
7assignvariableop_49_rmsprop_separable_conv2d_4_bias_rms?
;assignvariableop_50_rmsprop_batch_normalization_4_gamma_rms>
:assignvariableop_51_rmsprop_batch_normalization_4_beta_rmsG
Cassignvariableop_52_rmsprop_separable_conv2d_5_depthwise_kernel_rmsG
Cassignvariableop_53_rmsprop_separable_conv2d_5_pointwise_kernel_rms;
7assignvariableop_54_rmsprop_separable_conv2d_5_bias_rms?
;assignvariableop_55_rmsprop_batch_normalization_5_gamma_rms>
:assignvariableop_56_rmsprop_batch_normalization_5_beta_rmsG
Cassignvariableop_57_rmsprop_separable_conv2d_6_depthwise_kernel_rmsG
Cassignvariableop_58_rmsprop_separable_conv2d_6_pointwise_kernel_rms;
7assignvariableop_59_rmsprop_separable_conv2d_6_bias_rms?
;assignvariableop_60_rmsprop_batch_normalization_6_gamma_rms>
:assignvariableop_61_rmsprop_batch_normalization_6_beta_rmsG
Cassignvariableop_62_rmsprop_separable_conv2d_7_depthwise_kernel_rmsG
Cassignvariableop_63_rmsprop_separable_conv2d_7_pointwise_kernel_rms;
7assignvariableop_64_rmsprop_separable_conv2d_7_bias_rms?
;assignvariableop_65_rmsprop_batch_normalization_7_gamma_rms>
:assignvariableop_66_rmsprop_batch_normalization_7_beta_rms2
.assignvariableop_67_rmsprop_dense_4_kernel_rms0
,assignvariableop_68_rmsprop_dense_4_bias_rms2
.assignvariableop_69_rmsprop_dense_5_kernel_rms0
,assignvariableop_70_rmsprop_dense_5_bias_rms2
.assignvariableop_71_rmsprop_dense_6_kernel_rms0
,assignvariableop_72_rmsprop_dense_6_bias_rms2
.assignvariableop_73_rmsprop_dense_7_kernel_rms0
,assignvariableop_74_rmsprop_dense_7_bias_rms
identity_76ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1╦)
RestoreV2/tensor_namesConst"/device:CPU:0*ы(
valueу(BС(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:KЅ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:K*Ф
valueАBъKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ў
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*┬
_output_shapes»
г:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Y
dtypesO
M2K	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:|
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0ђ
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:ќ
AssignVariableOp_2AssignVariableOp6assignvariableop_2_separable_conv2d_4_depthwise_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:ќ
AssignVariableOp_3AssignVariableOp6assignvariableop_3_separable_conv2d_4_pointwise_kernelIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:і
AssignVariableOp_4AssignVariableOp*assignvariableop_4_separable_conv2d_4_biasIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:ј
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_4_gammaIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Ї
AssignVariableOp_6AssignVariableOp-assignvariableop_6_batch_normalization_4_betaIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
_output_shapes
:*
T0ћ
AssignVariableOp_7AssignVariableOp4assignvariableop_7_batch_normalization_4_moving_meanIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:ў
AssignVariableOp_8AssignVariableOp8assignvariableop_8_batch_normalization_4_moving_varianceIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
_output_shapes
:*
T0ќ
AssignVariableOp_9AssignVariableOp6assignvariableop_9_separable_conv2d_5_depthwise_kernelIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:Ў
AssignVariableOp_10AssignVariableOp7assignvariableop_10_separable_conv2d_5_pointwise_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
_output_shapes
:*
T0Ї
AssignVariableOp_11AssignVariableOp+assignvariableop_11_separable_conv2d_5_biasIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:Љ
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_5_gammaIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0љ
AssignVariableOp_13AssignVariableOp.assignvariableop_13_batch_normalization_5_betaIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:Ќ
AssignVariableOp_14AssignVariableOp5assignvariableop_14_batch_normalization_5_moving_meanIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:Џ
AssignVariableOp_15AssignVariableOp9assignvariableop_15_batch_normalization_5_moving_varianceIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Ў
AssignVariableOp_16AssignVariableOp7assignvariableop_16_separable_conv2d_6_depthwise_kernelIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Ў
AssignVariableOp_17AssignVariableOp7assignvariableop_17_separable_conv2d_6_pointwise_kernelIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:Ї
AssignVariableOp_18AssignVariableOp+assignvariableop_18_separable_conv2d_6_biasIdentity_18:output:0*
_output_shapes
 *
dtype0P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:Љ
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_6_gammaIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:љ
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batch_normalization_6_betaIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
_output_shapes
:*
T0Ќ
AssignVariableOp_21AssignVariableOp5assignvariableop_21_batch_normalization_6_moving_meanIdentity_21:output:0*
dtype0*
_output_shapes
 P
Identity_22IdentityRestoreV2:tensors:22*
_output_shapes
:*
T0Џ
AssignVariableOp_22AssignVariableOp9assignvariableop_22_batch_normalization_6_moving_varianceIdentity_22:output:0*
dtype0*
_output_shapes
 P
Identity_23IdentityRestoreV2:tensors:23*
_output_shapes
:*
T0Ў
AssignVariableOp_23AssignVariableOp7assignvariableop_23_separable_conv2d_7_depthwise_kernelIdentity_23:output:0*
dtype0*
_output_shapes
 P
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:Ў
AssignVariableOp_24AssignVariableOp7assignvariableop_24_separable_conv2d_7_pointwise_kernelIdentity_24:output:0*
dtype0*
_output_shapes
 P
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:Ї
AssignVariableOp_25AssignVariableOp+assignvariableop_25_separable_conv2d_7_biasIdentity_25:output:0*
dtype0*
_output_shapes
 P
Identity_26IdentityRestoreV2:tensors:26*
_output_shapes
:*
T0Љ
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_7_gammaIdentity_26:output:0*
_output_shapes
 *
dtype0P
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:љ
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_7_betaIdentity_27:output:0*
dtype0*
_output_shapes
 P
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:Ќ
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_7_moving_meanIdentity_28:output:0*
dtype0*
_output_shapes
 P
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:Џ
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_7_moving_varianceIdentity_29:output:0*
dtype0*
_output_shapes
 P
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:ё
AssignVariableOp_30AssignVariableOp"assignvariableop_30_dense_4_kernelIdentity_30:output:0*
dtype0*
_output_shapes
 P
Identity_31IdentityRestoreV2:tensors:31*
_output_shapes
:*
T0ѓ
AssignVariableOp_31AssignVariableOp assignvariableop_31_dense_4_biasIdentity_31:output:0*
dtype0*
_output_shapes
 P
Identity_32IdentityRestoreV2:tensors:32*
_output_shapes
:*
T0ё
AssignVariableOp_32AssignVariableOp"assignvariableop_32_dense_5_kernelIdentity_32:output:0*
dtype0*
_output_shapes
 P
Identity_33IdentityRestoreV2:tensors:33*
_output_shapes
:*
T0ѓ
AssignVariableOp_33AssignVariableOp assignvariableop_33_dense_5_biasIdentity_33:output:0*
dtype0*
_output_shapes
 P
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:ё
AssignVariableOp_34AssignVariableOp"assignvariableop_34_dense_6_kernelIdentity_34:output:0*
_output_shapes
 *
dtype0P
Identity_35IdentityRestoreV2:tensors:35*
_output_shapes
:*
T0ѓ
AssignVariableOp_35AssignVariableOp assignvariableop_35_dense_6_biasIdentity_35:output:0*
dtype0*
_output_shapes
 P
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:ё
AssignVariableOp_36AssignVariableOp"assignvariableop_36_dense_7_kernelIdentity_36:output:0*
dtype0*
_output_shapes
 P
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:ѓ
AssignVariableOp_37AssignVariableOp assignvariableop_37_dense_7_biasIdentity_37:output:0*
dtype0*
_output_shapes
 P
Identity_38IdentityRestoreV2:tensors:38*
_output_shapes
:*
T0	ѓ
AssignVariableOp_38AssignVariableOp assignvariableop_38_rmsprop_iterIdentity_38:output:0*
_output_shapes
 *
dtype0	P
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:Ѓ
AssignVariableOp_39AssignVariableOp!assignvariableop_39_rmsprop_decayIdentity_39:output:0*
dtype0*
_output_shapes
 P
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:І
AssignVariableOp_40AssignVariableOp)assignvariableop_40_rmsprop_learning_rateIdentity_40:output:0*
dtype0*
_output_shapes
 P
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:є
AssignVariableOp_41AssignVariableOp$assignvariableop_41_rmsprop_momentumIdentity_41:output:0*
dtype0*
_output_shapes
 P
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:Ђ
AssignVariableOp_42AssignVariableOpassignvariableop_42_rmsprop_rhoIdentity_42:output:0*
dtype0*
_output_shapes
 P
Identity_43IdentityRestoreV2:tensors:43*
_output_shapes
:*
T0{
AssignVariableOp_43AssignVariableOpassignvariableop_43_totalIdentity_43:output:0*
dtype0*
_output_shapes
 P
Identity_44IdentityRestoreV2:tensors:44*
_output_shapes
:*
T0{
AssignVariableOp_44AssignVariableOpassignvariableop_44_countIdentity_44:output:0*
dtype0*
_output_shapes
 P
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:Љ
AssignVariableOp_45AssignVariableOp/assignvariableop_45_rmsprop_conv2d_1_kernel_rmsIdentity_45:output:0*
dtype0*
_output_shapes
 P
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:Ј
AssignVariableOp_46AssignVariableOp-assignvariableop_46_rmsprop_conv2d_1_bias_rmsIdentity_46:output:0*
dtype0*
_output_shapes
 P
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:Ц
AssignVariableOp_47AssignVariableOpCassignvariableop_47_rmsprop_separable_conv2d_4_depthwise_kernel_rmsIdentity_47:output:0*
_output_shapes
 *
dtype0P
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:Ц
AssignVariableOp_48AssignVariableOpCassignvariableop_48_rmsprop_separable_conv2d_4_pointwise_kernel_rmsIdentity_48:output:0*
dtype0*
_output_shapes
 P
Identity_49IdentityRestoreV2:tensors:49*
_output_shapes
:*
T0Ў
AssignVariableOp_49AssignVariableOp7assignvariableop_49_rmsprop_separable_conv2d_4_bias_rmsIdentity_49:output:0*
dtype0*
_output_shapes
 P
Identity_50IdentityRestoreV2:tensors:50*
_output_shapes
:*
T0Ю
AssignVariableOp_50AssignVariableOp;assignvariableop_50_rmsprop_batch_normalization_4_gamma_rmsIdentity_50:output:0*
_output_shapes
 *
dtype0P
Identity_51IdentityRestoreV2:tensors:51*
_output_shapes
:*
T0ю
AssignVariableOp_51AssignVariableOp:assignvariableop_51_rmsprop_batch_normalization_4_beta_rmsIdentity_51:output:0*
_output_shapes
 *
dtype0P
Identity_52IdentityRestoreV2:tensors:52*
_output_shapes
:*
T0Ц
AssignVariableOp_52AssignVariableOpCassignvariableop_52_rmsprop_separable_conv2d_5_depthwise_kernel_rmsIdentity_52:output:0*
dtype0*
_output_shapes
 P
Identity_53IdentityRestoreV2:tensors:53*
_output_shapes
:*
T0Ц
AssignVariableOp_53AssignVariableOpCassignvariableop_53_rmsprop_separable_conv2d_5_pointwise_kernel_rmsIdentity_53:output:0*
dtype0*
_output_shapes
 P
Identity_54IdentityRestoreV2:tensors:54*
_output_shapes
:*
T0Ў
AssignVariableOp_54AssignVariableOp7assignvariableop_54_rmsprop_separable_conv2d_5_bias_rmsIdentity_54:output:0*
dtype0*
_output_shapes
 P
Identity_55IdentityRestoreV2:tensors:55*
T0*
_output_shapes
:Ю
AssignVariableOp_55AssignVariableOp;assignvariableop_55_rmsprop_batch_normalization_5_gamma_rmsIdentity_55:output:0*
dtype0*
_output_shapes
 P
Identity_56IdentityRestoreV2:tensors:56*
T0*
_output_shapes
:ю
AssignVariableOp_56AssignVariableOp:assignvariableop_56_rmsprop_batch_normalization_5_beta_rmsIdentity_56:output:0*
dtype0*
_output_shapes
 P
Identity_57IdentityRestoreV2:tensors:57*
T0*
_output_shapes
:Ц
AssignVariableOp_57AssignVariableOpCassignvariableop_57_rmsprop_separable_conv2d_6_depthwise_kernel_rmsIdentity_57:output:0*
dtype0*
_output_shapes
 P
Identity_58IdentityRestoreV2:tensors:58*
T0*
_output_shapes
:Ц
AssignVariableOp_58AssignVariableOpCassignvariableop_58_rmsprop_separable_conv2d_6_pointwise_kernel_rmsIdentity_58:output:0*
dtype0*
_output_shapes
 P
Identity_59IdentityRestoreV2:tensors:59*
T0*
_output_shapes
:Ў
AssignVariableOp_59AssignVariableOp7assignvariableop_59_rmsprop_separable_conv2d_6_bias_rmsIdentity_59:output:0*
dtype0*
_output_shapes
 P
Identity_60IdentityRestoreV2:tensors:60*
T0*
_output_shapes
:Ю
AssignVariableOp_60AssignVariableOp;assignvariableop_60_rmsprop_batch_normalization_6_gamma_rmsIdentity_60:output:0*
_output_shapes
 *
dtype0P
Identity_61IdentityRestoreV2:tensors:61*
T0*
_output_shapes
:ю
AssignVariableOp_61AssignVariableOp:assignvariableop_61_rmsprop_batch_normalization_6_beta_rmsIdentity_61:output:0*
dtype0*
_output_shapes
 P
Identity_62IdentityRestoreV2:tensors:62*
_output_shapes
:*
T0Ц
AssignVariableOp_62AssignVariableOpCassignvariableop_62_rmsprop_separable_conv2d_7_depthwise_kernel_rmsIdentity_62:output:0*
dtype0*
_output_shapes
 P
Identity_63IdentityRestoreV2:tensors:63*
T0*
_output_shapes
:Ц
AssignVariableOp_63AssignVariableOpCassignvariableop_63_rmsprop_separable_conv2d_7_pointwise_kernel_rmsIdentity_63:output:0*
dtype0*
_output_shapes
 P
Identity_64IdentityRestoreV2:tensors:64*
T0*
_output_shapes
:Ў
AssignVariableOp_64AssignVariableOp7assignvariableop_64_rmsprop_separable_conv2d_7_bias_rmsIdentity_64:output:0*
_output_shapes
 *
dtype0P
Identity_65IdentityRestoreV2:tensors:65*
_output_shapes
:*
T0Ю
AssignVariableOp_65AssignVariableOp;assignvariableop_65_rmsprop_batch_normalization_7_gamma_rmsIdentity_65:output:0*
dtype0*
_output_shapes
 P
Identity_66IdentityRestoreV2:tensors:66*
T0*
_output_shapes
:ю
AssignVariableOp_66AssignVariableOp:assignvariableop_66_rmsprop_batch_normalization_7_beta_rmsIdentity_66:output:0*
dtype0*
_output_shapes
 P
Identity_67IdentityRestoreV2:tensors:67*
T0*
_output_shapes
:љ
AssignVariableOp_67AssignVariableOp.assignvariableop_67_rmsprop_dense_4_kernel_rmsIdentity_67:output:0*
dtype0*
_output_shapes
 P
Identity_68IdentityRestoreV2:tensors:68*
_output_shapes
:*
T0ј
AssignVariableOp_68AssignVariableOp,assignvariableop_68_rmsprop_dense_4_bias_rmsIdentity_68:output:0*
dtype0*
_output_shapes
 P
Identity_69IdentityRestoreV2:tensors:69*
T0*
_output_shapes
:љ
AssignVariableOp_69AssignVariableOp.assignvariableop_69_rmsprop_dense_5_kernel_rmsIdentity_69:output:0*
dtype0*
_output_shapes
 P
Identity_70IdentityRestoreV2:tensors:70*
T0*
_output_shapes
:ј
AssignVariableOp_70AssignVariableOp,assignvariableop_70_rmsprop_dense_5_bias_rmsIdentity_70:output:0*
dtype0*
_output_shapes
 P
Identity_71IdentityRestoreV2:tensors:71*
T0*
_output_shapes
:љ
AssignVariableOp_71AssignVariableOp.assignvariableop_71_rmsprop_dense_6_kernel_rmsIdentity_71:output:0*
dtype0*
_output_shapes
 P
Identity_72IdentityRestoreV2:tensors:72*
T0*
_output_shapes
:ј
AssignVariableOp_72AssignVariableOp,assignvariableop_72_rmsprop_dense_6_bias_rmsIdentity_72:output:0*
dtype0*
_output_shapes
 P
Identity_73IdentityRestoreV2:tensors:73*
_output_shapes
:*
T0љ
AssignVariableOp_73AssignVariableOp.assignvariableop_73_rmsprop_dense_7_kernel_rmsIdentity_73:output:0*
dtype0*
_output_shapes
 P
Identity_74IdentityRestoreV2:tensors:74*
T0*
_output_shapes
:ј
AssignVariableOp_74AssignVariableOp,assignvariableop_74_rmsprop_dense_7_bias_rmsIdentity_74:output:0*
dtype0*
_output_shapes
 ї
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 ┴
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ╬
Identity_76IdentityIdentity_75:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_76Identity_76:output:0*├
_input_shapes▒
«: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_19AssignVariableOp_192*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322$
AssignVariableOpAssignVariableOp2*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_39AssignVariableOp_392*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_49AssignVariableOp_492*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_59AssignVariableOp_592*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692*
AssignVariableOp_74AssignVariableOp_74: : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< := :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : 
А
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656887

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:         ђ*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:         ђ*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
­x
ш
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657417

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_45
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_45
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_45
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб*separable_conv2d_4/StatefulPartitionedCallб*separable_conv2d_5/StatefulPartitionedCallб*separable_conv2d_6/StatefulPartitionedCallб*separable_conv2d_7/StatefulPartitionedCallћ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2655672*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         ddр
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2655691*U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         22*
Tin
2њ
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*.
_gradient_op_typePartitionedCall-2655719*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         22 с
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         22 *.
_gradient_op_typePartitionedCall-2656538*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656525Ь
max_pooling2d_6/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*/
_output_shapes
:          *
Tin
2*.
_gradient_op_typePartitionedCall-2655881*U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875*
Tout
2*-
config_proto

GPU

CPU2*0J 8њ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2655909*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903*
Tout
2*-
config_proto

GPU

CPU2*0J 8с
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2656626*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656613*
Tout
2Ь
max_pooling2d_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656071*U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         @*
Tin
2Њ
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656099*X
fSRQ
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093*
Tout
2С
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656714*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656701*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ№
max_pooling2d_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656261*U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255*
Tout
2Н
dropout_5/PartitionedCallPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656767*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656755*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђЇ
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*.
_gradient_op_typePartitionedCall-2656289*X
fSRQ
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђС
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656846*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656833*
Tout
2№
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656451*U
fPRN
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2Н
dropout_6/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656899*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656887*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђК
flatten_1/PartitionedCallPartitionedCall"dropout_6/PartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656916*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910Ц
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656940*M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934*
Tout
2═
dropout_7/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656990*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656978*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2Ц
dense_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_7/PartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2657012*M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006*
Tout
2*-
config_proto

GPU

CPU2*0J 8═
dropout_8/PartitionedCallPartitionedCall(dense_5/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2657062*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657050*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђц
dense_6/StatefulPartitionedCallStatefulPartitionedCall"dropout_8/PartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657084*M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         @*
Tin
2╠
dropout_9/PartitionedCallPartitionedCall(dense_6/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2657134*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657122*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         @ц
dense_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_9/PartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-2657156*M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150*
Tout
2Ј
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
С
ш
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656833

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ╗
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ¤
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
Љ
┘
4__inference_separable_conv2d_6_layer_call_fn_2656105

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*B
_output_shapes0
.:,                           ђ*.
_gradient_op_typePartitionedCall-2656099*X
fSRQ
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*L
_input_shapes;
9:+                           @:::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
░/
Ќ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658179

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: └
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
: *
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         22 "
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
▄
d
+__inference_dropout_6_layer_call_fn_2658858

inputs
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-2656891*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656880*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
С
ш
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656701

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0╗
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ¤
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs
щ/
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656207

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђ*
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ч
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs: : : : 
щ/
Ќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656397

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0w
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
T0Р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0ч
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
п
G
+__inference_dropout_5_layer_call_fn_2658662

inputs
identityЕ
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656767*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656755*
Tout
2i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
▒
ђ
7__inference_batch_normalization_6_layer_call_fn_2658542

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656704*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656679І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : 
џ
ш
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658810

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0Ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: р
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
г
Ф
*__inference_conv2d_1_layer_call_fn_2655677

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*A
_output_shapes/
-:+                           *
Tin
2*.
_gradient_op_typePartitionedCall-2655672*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           *
T0"
identityIdentity:output:0*H
_input_shapes7
5:+                           ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
А
d
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658853

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:         ђ*
T0d

Identity_1IdentityIdentity:output:0*0
_output_shapes
:         ђ*
T0"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
│
e
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658912

inputs
identityѕQ
dropout/rateConst*
valueB
 *џЎЎ>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         ђЋ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:         ђ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         ђ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
└
G
+__inference_dropout_7_layer_call_fn_2658927

inputs
identityА
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-2656990*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656978*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђa
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
├/
Ќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658712

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
dtype0*
_output_shapes
: *
valueB J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0ж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: :& "
 
_user_specified_nameinputs: : : 
▒
ђ
7__inference_batch_normalization_7_layer_call_fn_2658743

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656836*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656811*
Tout
2І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
у
ђ
7__inference_batch_normalization_7_layer_call_fn_2658828

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656431*
Tout
2*-
config_proto

GPU

CPU2*0J 8*B
_output_shapes0
.:,                           ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656432Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
й
G
+__inference_dropout_9_layer_call_fn_2659033

inputs
identityа
PartitionedCallPartitionedCallinputs*.
_gradient_op_typePartitionedCall-2657134*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657122*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         @*
Tin
2`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*&
_input_shapes
:         @:& "
 
_user_specified_nameinputs
Дњ
╦&
"__inference__wrapped_model_2655652
conv2d_1_input8
4sequential_1_conv2d_1_conv2d_readvariableop_resource9
5sequential_1_conv2d_1_biasadd_readvariableop_resourceL
Hsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_4_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_4_readvariableop_resource@
<sequential_1_batch_normalization_4_readvariableop_1_resourceO
Ksequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceL
Hsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_5_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_5_readvariableop_resource@
<sequential_1_batch_normalization_5_readvariableop_1_resourceO
Ksequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceL
Hsequential_1_separable_conv2d_6_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_6_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_6_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_6_readvariableop_resource@
<sequential_1_batch_normalization_6_readvariableop_1_resourceO
Ksequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceL
Hsequential_1_separable_conv2d_7_separable_conv2d_readvariableop_resourceN
Jsequential_1_separable_conv2d_7_separable_conv2d_readvariableop_1_resourceC
?sequential_1_separable_conv2d_7_biasadd_readvariableop_resource>
:sequential_1_batch_normalization_7_readvariableop_resource@
<sequential_1_batch_normalization_7_readvariableop_1_resourceO
Ksequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceQ
Msequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource7
3sequential_1_dense_4_matmul_readvariableop_resource8
4sequential_1_dense_4_biasadd_readvariableop_resource7
3sequential_1_dense_5_matmul_readvariableop_resource8
4sequential_1_dense_5_biasadd_readvariableop_resource7
3sequential_1_dense_6_matmul_readvariableop_resource8
4sequential_1_dense_6_biasadd_readvariableop_resource7
3sequential_1_dense_7_matmul_readvariableop_resource8
4sequential_1_dense_7_biasadd_readvariableop_resource
identityѕбBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_4/ReadVariableOpб3sequential_1/batch_normalization_4/ReadVariableOp_1бBsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_5/ReadVariableOpб3sequential_1/batch_normalization_5/ReadVariableOp_1бBsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_6/ReadVariableOpб3sequential_1/batch_normalization_6/ReadVariableOp_1бBsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpбDsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б1sequential_1/batch_normalization_7/ReadVariableOpб3sequential_1/batch_normalization_7/ReadVariableOp_1б,sequential_1/conv2d_1/BiasAdd/ReadVariableOpб+sequential_1/conv2d_1/Conv2D/ReadVariableOpб+sequential_1/dense_4/BiasAdd/ReadVariableOpб*sequential_1/dense_4/MatMul/ReadVariableOpб+sequential_1/dense_5/BiasAdd/ReadVariableOpб*sequential_1/dense_5/MatMul/ReadVariableOpб+sequential_1/dense_6/BiasAdd/ReadVariableOpб*sequential_1/dense_6/MatMul/ReadVariableOpб+sequential_1/dense_7/BiasAdd/ReadVariableOpб*sequential_1/dense_7/MatMul/ReadVariableOpб6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOpб?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOpбAsequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1б6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOpб?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOpбAsequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1б6sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOpб?sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOpбAsequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_1б6sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOpб?sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOpбAsequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_1о
+sequential_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4sequential_1_conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:═
sequential_1/conv2d_1/Conv2DConv2Dconv2d_1_input3sequential_1/conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         dd*
T0*
strides
*
paddingSAME╠
,sequential_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5sequential_1_conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:┐
sequential_1/conv2d_1/BiasAddBiasAdd%sequential_1/conv2d_1/Conv2D:output:04sequential_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         dd*
T0ё
sequential_1/conv2d_1/ReluRelu&sequential_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         dd*
T0к
$sequential_1/max_pooling2d_5/MaxPoolMaxPool(sequential_1/conv2d_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:         22■
?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:ѓ
Asequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_4_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: Ј
6sequential_1/separable_conv2d_4/separable_conv2d/ShapeConst*%
valueB"            *
dtype0*
_output_shapes
:Ј
>sequential_1/separable_conv2d_4/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:Г
:sequential_1/separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative-sequential_1/max_pooling2d_5/MaxPool:output:0Gsequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:         22*
T0Г
0sequential_1/separable_conv2d_4/separable_conv2dConv2DCsequential_1/separable_conv2d_4/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         22 Я
6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0у
'sequential_1/separable_conv2d_4/BiasAddBiasAdd9sequential_1/separable_conv2d_4/separable_conv2d:output:0>sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         22 *
T0ў
$sequential_1/separable_conv2d_4/ReluRelu0sequential_1/separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         22 q
/sequential_1/batch_normalization_4/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/sequential_1/batch_normalization_4/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 ZК
-sequential_1/batch_normalization_4/LogicalAnd
LogicalAnd8sequential_1/batch_normalization_4/LogicalAnd/x:output:08sequential_1/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: о
1sequential_1/batch_normalization_4/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_4_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ┌
3sequential_1/batch_normalization_4/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_4_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Э
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ч
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Љ
3sequential_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV32sequential_1/separable_conv2d_4/Relu:activations:09sequential_1/batch_normalization_4/ReadVariableOp:value:0;sequential_1/batch_normalization_4/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         22 : : : : :*
T0*
U0*
is_training( *
epsilon%oЃ:m
(sequential_1/batch_normalization_4/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *цp}?Н
$sequential_1/max_pooling2d_6/MaxPoolMaxPool7sequential_1/batch_normalization_4/FusedBatchNormV3:y:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:          ■
?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ѓ
Asequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_5_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @Ј
6sequential_1/separable_conv2d_5/separable_conv2d/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:Ј
>sequential_1/separable_conv2d_5/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:Г
:sequential_1/separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative-sequential_1/max_pooling2d_6/MaxPool:output:0Gsequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*/
_output_shapes
:          *
T0*
strides
*
paddingSAMEГ
0sequential_1/separable_conv2d_5/separable_conv2dConv2DCsequential_1/separable_conv2d_5/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         @Я
6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@у
'sequential_1/separable_conv2d_5/BiasAddBiasAdd9sequential_1/separable_conv2d_5/separable_conv2d:output:0>sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ў
$sequential_1/separable_conv2d_5/ReluRelu0sequential_1/separable_conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @q
/sequential_1/batch_normalization_5/LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z q
/sequential_1/batch_normalization_5/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 ZК
-sequential_1/batch_normalization_5/LogicalAnd
LogicalAnd8sequential_1/batch_normalization_5/LogicalAnd/x:output:08sequential_1/batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: о
1sequential_1/batch_normalization_5/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_5_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0┌
3sequential_1/batch_normalization_5/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_5_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
Bsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ч
Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Љ
3sequential_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV32sequential_1/separable_conv2d_5/Relu:activations:09sequential_1/batch_normalization_5/ReadVariableOp:value:0;sequential_1/batch_normalization_5/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*K
_output_shapes9
7:         @:@:@:@:@:*
T0*
U0*
is_training( m
(sequential_1/batch_normalization_5/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Н
$sequential_1/max_pooling2d_7/MaxPoolMaxPool7sequential_1/batch_normalization_5/FusedBatchNormV3:y:0*
paddingVALID*/
_output_shapes
:         @*
strides
*
ksize
■
?sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_6_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@Ѓ
Asequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_6_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@ђЈ
6sequential_1/separable_conv2d_6/separable_conv2d/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:Ј
>sequential_1/separable_conv2d_6/separable_conv2d/dilation_rateConst*
_output_shapes
:*
valueB"      *
dtype0Г
:sequential_1/separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNative-sequential_1/max_pooling2d_7/MaxPool:output:0Gsequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:         @«
0sequential_1/separable_conv2d_6/separable_conv2dConv2DCsequential_1/separable_conv2d_6/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*0
_output_shapes
:         ђ*
T0*
strides
*
paddingVALIDр
6sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђУ
'sequential_1/separable_conv2d_6/BiasAddBiasAdd9sequential_1/separable_conv2d_6/separable_conv2d:output:0>sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЎ
$sequential_1/separable_conv2d_6/ReluRelu0sequential_1/separable_conv2d_6/BiasAdd:output:0*0
_output_shapes
:         ђ*
T0q
/sequential_1/batch_normalization_6/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/sequential_1/batch_normalization_6/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: К
-sequential_1/batch_normalization_6/LogicalAnd
LogicalAnd8sequential_1/batch_normalization_6/LogicalAnd/x:output:08sequential_1/batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: О
1sequential_1/batch_normalization_6/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_6_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0█
3sequential_1/batch_normalization_6/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_6_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
Bsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ§
Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђќ
3sequential_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV32sequential_1/separable_conv2d_6/Relu:activations:09sequential_1/batch_normalization_6/ReadVariableOp:value:0;sequential_1/batch_normalization_6/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:m
(sequential_1/batch_normalization_6/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: о
$sequential_1/max_pooling2d_8/MaxPoolMaxPool7sequential_1/batch_normalization_6/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
strides
*
ksize
*
paddingVALIDЋ
sequential_1/dropout_5/IdentityIdentity-sequential_1/max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         ђ 
?sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOpHsequential_1_separable_conv2d_7_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:ђё
Asequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOpJsequential_1_separable_conv2d_7_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ђђЈ
6sequential_1/separable_conv2d_7/separable_conv2d/ShapeConst*%
valueB"      ђ      *
dtype0*
_output_shapes
:Ј
>sequential_1/separable_conv2d_7/separable_conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      Е
:sequential_1/separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNative(sequential_1/dropout_5/Identity:output:0Gsequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*0
_output_shapes
:         ђ*
T0*
strides
*
paddingSAME«
0sequential_1/separable_conv2d_7/separable_conv2dConv2DCsequential_1/separable_conv2d_7/separable_conv2d/depthwise:output:0Isequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
strides
*
paddingVALID*0
_output_shapes
:         ђ*
T0р
6sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp?sequential_1_separable_conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђУ
'sequential_1/separable_conv2d_7/BiasAddBiasAdd9sequential_1/separable_conv2d_7/separable_conv2d:output:0>sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђЎ
$sequential_1/separable_conv2d_7/ReluRelu0sequential_1/separable_conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         ђq
/sequential_1/batch_normalization_7/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/sequential_1/batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
К
-sequential_1/batch_normalization_7/LogicalAnd
LogicalAnd8sequential_1/batch_normalization_7/LogicalAnd/x:output:08sequential_1/batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: О
1sequential_1/batch_normalization_7/ReadVariableOpReadVariableOp:sequential_1_batch_normalization_7_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ█
3sequential_1/batch_normalization_7/ReadVariableOp_1ReadVariableOp<sequential_1_batch_normalization_7_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0щ
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpKsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ§
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMsequential_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђќ
3sequential_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV32sequential_1/separable_conv2d_7/Relu:activations:09sequential_1/batch_normalization_7/ReadVariableOp:value:0;sequential_1/batch_normalization_7/ReadVariableOp_1:value:0Jsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:m
(sequential_1/batch_normalization_7/ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0о
$sequential_1/max_pooling2d_9/MaxPoolMaxPool7sequential_1/batch_normalization_7/FusedBatchNormV3:y:0*
ksize
*
paddingVALID*0
_output_shapes
:         ђ*
strides
Ћ
sequential_1/dropout_6/IdentityIdentity-sequential_1/max_pooling2d_9/MaxPool:output:0*
T0*0
_output_shapes
:         ђu
$sequential_1/flatten_1/Reshape/shapeConst*
valueB"     	  *
dtype0*
_output_shapes
:х
sequential_1/flatten_1/ReshapeReshape(sequential_1/dropout_6/Identity:output:0-sequential_1/flatten_1/Reshape/shape:output:0*
T0*(
_output_shapes
:         ђ╬
*sequential_1/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђх
sequential_1/dense_4/MatMulMatMul'sequential_1/flatten_1/Reshape:output:02sequential_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╦
+sequential_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђХ
sequential_1/dense_4/BiasAddBiasAdd%sequential_1/dense_4/MatMul:product:03sequential_1/dense_4/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0{
sequential_1/dense_4/ReluRelu%sequential_1/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         ђЄ
sequential_1/dropout_7/IdentityIdentity'sequential_1/dense_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ╬
*sequential_1/dense_5/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђХ
sequential_1/dense_5/MatMulMatMul(sequential_1/dropout_7/Identity:output:02sequential_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╦
+sequential_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђХ
sequential_1/dense_5/BiasAddBiasAdd%sequential_1/dense_5/MatMul:product:03sequential_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ{
sequential_1/dense_5/ReluRelu%sequential_1/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:         ђЄ
sequential_1/dropout_8/IdentityIdentity'sequential_1/dense_5/Relu:activations:0*(
_output_shapes
:         ђ*
T0═
*sequential_1/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	ђ@х
sequential_1/dense_6/MatMulMatMul(sequential_1/dropout_8/Identity:output:02sequential_1/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @╩
+sequential_1/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@х
sequential_1/dense_6/BiasAddBiasAdd%sequential_1/dense_6/MatMul:product:03sequential_1/dense_6/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0z
sequential_1/dense_6/ReluRelu%sequential_1/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         @є
sequential_1/dropout_9/IdentityIdentity'sequential_1/dense_6/Relu:activations:0*
T0*'
_output_shapes
:         @╠
*sequential_1/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@х
sequential_1/dense_7/MatMulMatMul(sequential_1/dropout_9/Identity:output:02sequential_1/dense_7/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0╩
+sequential_1/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:х
sequential_1/dense_7/BiasAddBiasAdd%sequential_1/dense_7/MatMul:product:03sequential_1/dense_7/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0ђ
sequential_1/dense_7/SigmoidSigmoid%sequential_1/dense_7/BiasAdd:output:0*'
_output_shapes
:         *
T0Ё
IdentityIdentity sequential_1/dense_7/Sigmoid:y:0C^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_4/ReadVariableOp4^sequential_1/batch_normalization_4/ReadVariableOp_1C^sequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_5/ReadVariableOp4^sequential_1/batch_normalization_5/ReadVariableOp_1C^sequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_6/ReadVariableOp4^sequential_1/batch_normalization_6/ReadVariableOp_1C^sequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpE^sequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12^sequential_1/batch_normalization_7/ReadVariableOp4^sequential_1/batch_normalization_7/ReadVariableOp_1-^sequential_1/conv2d_1/BiasAdd/ReadVariableOp,^sequential_1/conv2d_1/Conv2D/ReadVariableOp,^sequential_1/dense_4/BiasAdd/ReadVariableOp+^sequential_1/dense_4/MatMul/ReadVariableOp,^sequential_1/dense_5/BiasAdd/ReadVariableOp+^sequential_1/dense_5/MatMul/ReadVariableOp,^sequential_1/dense_6/BiasAdd/ReadVariableOp+^sequential_1/dense_6/MatMul/ReadVariableOp,^sequential_1/dense_7/BiasAdd/ReadVariableOp+^sequential_1/dense_7/MatMul/ReadVariableOp7^sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp@^sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOpB^sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_17^sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp@^sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOpB^sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_17^sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOp@^sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOpB^sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_17^sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOp@^sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOpB^sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_1*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2є
Asequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_1Asequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp_12X
*sequential_1/dense_5/MatMul/ReadVariableOp*sequential_1/dense_5/MatMul/ReadVariableOp2f
1sequential_1/batch_normalization_4/ReadVariableOp1sequential_1/batch_normalization_4/ReadVariableOp2є
Asequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_1Asequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp_12p
6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp6sequential_1/separable_conv2d_4/BiasAdd/ReadVariableOp2Z
+sequential_1/dense_5/BiasAdd/ReadVariableOp+sequential_1/dense_5/BiasAdd/ReadVariableOp2X
*sequential_1/dense_6/MatMul/ReadVariableOp*sequential_1/dense_6/MatMul/ReadVariableOp2ѓ
?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp?sequential_1/separable_conv2d_4/separable_conv2d/ReadVariableOp2ѓ
?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp?sequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp2p
6sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOp6sequential_1/separable_conv2d_7/BiasAdd/ReadVariableOp2f
1sequential_1/batch_normalization_5/ReadVariableOp1sequential_1/batch_normalization_5/ReadVariableOp2ѓ
?sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp?sequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp2ѓ
?sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp?sequential_1/separable_conv2d_7/separable_conv2d/ReadVariableOp2ї
Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12є
Asequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_1Asequential_1/separable_conv2d_5/separable_conv2d/ReadVariableOp_12X
*sequential_1/dense_7/MatMul/ReadVariableOp*sequential_1/dense_7/MatMul/ReadVariableOp2Z
+sequential_1/dense_4/BiasAdd/ReadVariableOp+sequential_1/dense_4/BiasAdd/ReadVariableOp2f
1sequential_1/batch_normalization_6/ReadVariableOp1sequential_1/batch_normalization_6/ReadVariableOp2Z
+sequential_1/conv2d_1/Conv2D/ReadVariableOp+sequential_1/conv2d_1/Conv2D/ReadVariableOp2ї
Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12p
6sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOp6sequential_1/separable_conv2d_6/BiasAdd/ReadVariableOp2\
,sequential_1/conv2d_1/BiasAdd/ReadVariableOp,sequential_1/conv2d_1/BiasAdd/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2Z
+sequential_1/dense_7/BiasAdd/ReadVariableOp+sequential_1/dense_7/BiasAdd/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2ѕ
Bsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpBsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2j
3sequential_1/batch_normalization_4/ReadVariableOp_13sequential_1/batch_normalization_4/ReadVariableOp_12f
1sequential_1/batch_normalization_7/ReadVariableOp1sequential_1/batch_normalization_7/ReadVariableOp2ї
Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12j
3sequential_1/batch_normalization_5/ReadVariableOp_13sequential_1/batch_normalization_5/ReadVariableOp_12є
Asequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_1Asequential_1/separable_conv2d_6/separable_conv2d/ReadVariableOp_12j
3sequential_1/batch_normalization_6/ReadVariableOp_13sequential_1/batch_normalization_6/ReadVariableOp_12p
6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp6sequential_1/separable_conv2d_5/BiasAdd/ReadVariableOp2j
3sequential_1/batch_normalization_7/ReadVariableOp_13sequential_1/batch_normalization_7/ReadVariableOp_12Z
+sequential_1/dense_6/BiasAdd/ReadVariableOp+sequential_1/dense_6/BiasAdd/ReadVariableOp2ї
Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Dsequential_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12X
*sequential_1/dense_4/MatMul/ReadVariableOp*sequential_1/dense_4/MatMul/ReadVariableOp:# :$ :% :& :. *
(
_user_specified_nameconv2d_1_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" 
Т/
Ќ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2655827

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                            : : : : :*
T0L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: └
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpМ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
: *
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs: : : : 
э█
Ш&
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657872

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceF
Bbatch_normalization_4_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_4_assignmovingavg_1_read_readvariableop_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceF
Bbatch_normalization_5_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_5_assignmovingavg_1_read_readvariableop_resource?
;separable_conv2d_6_separable_conv2d_readvariableop_resourceA
=separable_conv2d_6_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceF
Bbatch_normalization_6_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_6_assignmovingavg_1_read_readvariableop_resource?
;separable_conv2d_7_separable_conv2d_readvariableop_resourceA
=separable_conv2d_7_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceF
Bbatch_normalization_7_assignmovingavg_read_readvariableop_resourceH
Dbatch_normalization_7_assignmovingavg_1_read_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕб9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpб9batch_normalization_4/AssignMovingAvg/Read/ReadVariableOpб4batch_normalization_4/AssignMovingAvg/ReadVariableOpб;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpб;batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOpб6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpб$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpб9batch_normalization_5/AssignMovingAvg/Read/ReadVariableOpб4batch_normalization_5/AssignMovingAvg/ReadVariableOpб;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpб;batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOpб6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpб$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpб9batch_normalization_6/AssignMovingAvg/Read/ReadVariableOpб4batch_normalization_6/AssignMovingAvg/ReadVariableOpб;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpб;batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOpб6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpб$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpб9batch_normalization_7/AssignMovingAvg/Read/ReadVariableOpб4batch_normalization_7/AssignMovingAvg/ReadVariableOpб;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpб;batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOpб6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpб$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOpб)separable_conv2d_4/BiasAdd/ReadVariableOpб2separable_conv2d_4/separable_conv2d/ReadVariableOpб4separable_conv2d_4/separable_conv2d/ReadVariableOp_1б)separable_conv2d_5/BiasAdd/ReadVariableOpб2separable_conv2d_5/separable_conv2d/ReadVariableOpб4separable_conv2d_5/separable_conv2d/ReadVariableOp_1б)separable_conv2d_6/BiasAdd/ReadVariableOpб2separable_conv2d_6/separable_conv2d/ReadVariableOpб4separable_conv2d_6/separable_conv2d/ReadVariableOp_1б)separable_conv2d_7/BiasAdd/ReadVariableOpб2separable_conv2d_7/separable_conv2d/ReadVariableOpб4separable_conv2d_7/separable_conv2d/ReadVariableOp_1╝
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:Ф
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:         dd*
T0*
strides
▓
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         dd*
T0j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         ddг
max_pooling2d_5/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
ksize
*
paddingVALID*/
_output_shapes
:         22*
strides
С
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:У
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ѓ
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0ѓ
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
dtype0*
_output_shapes
:*
valueB"      є
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_5/MaxPool:output:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*/
_output_shapes
:         22*
T0*
strides
*
paddingSAMEє
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         22 к
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: └
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         22 ~
separable_conv2d_4/ReluRelu#separable_conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:         22 d
"batch_normalization_4/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_4/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: ╝
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: └
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ^
batch_normalization_4/ConstConst*
valueB *
dtype0*
_output_shapes
: `
batch_normalization_4/Const_1Const*
_output_shapes
: *
valueB *
dtype0■
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0$batch_normalization_4/Const:output:0&batch_normalization_4/Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :b
batch_normalization_4/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: Т
9batch_normalization_4/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_4_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: б
.batch_normalization_4/AssignMovingAvg/IdentityIdentityAbatch_normalization_4/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
: *
T0В
+batch_normalization_4/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*L
_classB
@>loc:@batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: Ф
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0&batch_normalization_4/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Ю
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_4_assignmovingavg_read_readvariableop_resource:^batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: к
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0▒
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0Ў
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_4_assignmovingavg_read_readvariableop_resource-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ж
;batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_4_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: д
0batch_normalization_4/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: ­
-batch_normalization_4/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*N
_classD
B@loc:@batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ▒
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0&batch_normalization_4/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*N
_classD
B@loc:@batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOpБ
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_4_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: л
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: ╣
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Б
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_4_assignmovingavg_1_read_readvariableop_resource/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ╗
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:          С
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: У
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @ѓ
)separable_conv2d_5/separable_conv2d/ShapeConst*
_output_shapes
:*%
valueB"             *
dtype0ѓ
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:є
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_6/MaxPool:output:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
strides
*
paddingSAME*/
_output_shapes
:          *
T0є
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*/
_output_shapes
:         @*
T0*
strides
*
paddingVALIDк
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@└
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         @*
T0~
separable_conv2d_5/ReluRelu#separable_conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:         @d
"batch_normalization_5/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
а
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: ╝
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@└
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0^
batch_normalization_5/ConstConst*
valueB *
dtype0*
_output_shapes
: `
batch_normalization_5/Const_1Const*
_output_shapes
: *
valueB *
dtype0■
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0$batch_normalization_5/Const:output:0&batch_normalization_5/Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         @:@:@:@:@:*
T0*
U0b
batch_normalization_5/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: Т
9batch_normalization_5/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_5_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@б
.batch_normalization_5/AssignMovingAvg/IdentityIdentityAbatch_normalization_5/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@В
+batch_normalization_5/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*L
_classB
@>loc:@batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: Ф
)batch_normalization_5/AssignMovingAvg/subSub4batch_normalization_5/AssignMovingAvg/sub/x:output:0&batch_normalization_5/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: Ю
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_5_assignmovingavg_read_readvariableop_resource:^batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@к
+batch_normalization_5/AssignMovingAvg/sub_1Sub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_5/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@▒
)batch_normalization_5/AssignMovingAvg/mulMul/batch_normalization_5/AssignMovingAvg/sub_1:z:0-batch_normalization_5/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ў
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_5_assignmovingavg_read_readvariableop_resource-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ж
;batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_5_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@д
0batch_normalization_5/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@­
-batch_normalization_5/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*N
_classD
B@loc:@batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ▒
+batch_normalization_5/AssignMovingAvg_1/subSub6batch_normalization_5/AssignMovingAvg_1/sub/x:output:0&batch_normalization_5/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Б
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_5_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@л
-batch_normalization_5/AssignMovingAvg_1/sub_1Sub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_5/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0╣
+batch_normalization_5/AssignMovingAvg_1/mulMul1batch_normalization_5/AssignMovingAvg_1/sub_1:z:0/batch_normalization_5/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@Б
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_5_assignmovingavg_1_read_readvariableop_resource/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ╗
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*
ksize
*
paddingVALID*/
_output_shapes
:         @*
strides
С
2separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_6_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@*
dtype0ж
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_6_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*'
_output_shapes
:@ђ*
dtype0ѓ
)separable_conv2d_6/separable_conv2d/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:ѓ
1separable_conv2d_6/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:є
-separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_7/MaxPool:output:0:separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:         @*
T0*
strides
Є
#separable_conv2d_6/separable_conv2dConv2D6separable_conv2d_6/separable_conv2d/depthwise:output:0<separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*
strides
*
paddingVALID*0
_output_shapes
:         ђ*
T0К
)separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
separable_conv2d_6/BiasAddBiasAdd,separable_conv2d_6/separable_conv2d:output:01separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
separable_conv2d_6/ReluRelu#separable_conv2d_6/BiasAdd:output:0*0
_output_shapes
:         ђ*
T0d
"batch_normalization_6/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_6/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: й
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ^
batch_normalization_6/ConstConst*
valueB *
dtype0*
_output_shapes
: `
batch_normalization_6/Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0$batch_normalization_6/Const:output:0&batch_normalization_6/Const_1:output:0*
T0*
U0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:b
batch_normalization_6/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: у
9batch_normalization_6/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_6_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђБ
.batch_normalization_6/AssignMovingAvg/IdentityIdentityAbatch_normalization_6/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0В
+batch_normalization_6/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*L
_classB
@>loc:@batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: Ф
)batch_normalization_6/AssignMovingAvg/subSub4batch_normalization_6/AssignMovingAvg/sub/x:output:0&batch_normalization_6/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ъ
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_6_assignmovingavg_read_readvariableop_resource:^batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђК
+batch_normalization_6/AssignMovingAvg/sub_1Sub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_6/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*L
_classB
@>loc:@batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp▓
)batch_normalization_6/AssignMovingAvg/mulMul/batch_normalization_6/AssignMovingAvg/sub_1:z:0-batch_normalization_6/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђЎ
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_6_assignmovingavg_read_readvariableop_resource-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *L
_classB
@>loc:@batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp*
dtype0в
;batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_6_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђД
0batch_normalization_6/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ­
-batch_normalization_6/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*N
_classD
B@loc:@batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ▒
+batch_normalization_6/AssignMovingAvg_1/subSub6batch_normalization_6/AssignMovingAvg_1/sub/x:output:0&batch_normalization_6/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: ц
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_6_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЛ
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_6/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђ║
+batch_normalization_6/AssignMovingAvg_1/mulMul1batch_normalization_6/AssignMovingAvg_1/sub_1:z:0/batch_normalization_6/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђБ
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_6_assignmovingavg_1_read_readvariableop_resource/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ╝
max_pooling2d_8/MaxPoolMaxPool*batch_normalization_6/FusedBatchNormV3:y:0*0
_output_shapes
:         ђ*
strides
*
ksize
*
paddingVALID[
dropout_5/dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: g
dropout_5/dropout/ShapeShape max_pooling2d_8/MaxPool:output:0*
T0*
_output_shapes
:i
$dropout_5/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_5/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Е
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*0
_output_shapes
:         ђ*
T0*
dtype0ф
$dropout_5/dropout/random_uniform/subSub-dropout_5/dropout/random_uniform/max:output:0-dropout_5/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0╔
$dropout_5/dropout/random_uniform/mulMul7dropout_5/dropout/random_uniform/RandomUniform:output:0(dropout_5/dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         ђ╗
 dropout_5/dropout/random_uniformAdd(dropout_5/dropout/random_uniform/mul:z:0-dropout_5/dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         ђ\
dropout_5/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?ђ
dropout_5/dropout/subSub dropout_5/dropout/sub/x:output:0dropout_5/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_5/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?є
dropout_5/dropout/truedivRealDiv$dropout_5/dropout/truediv/x:output:0dropout_5/dropout/sub:z:0*
T0*
_output_shapes
: ░
dropout_5/dropout/GreaterEqualGreaterEqual$dropout_5/dropout/random_uniform:z:0dropout_5/dropout/rate:output:0*0
_output_shapes
:         ђ*
T0ў
dropout_5/dropout/mulMul max_pooling2d_8/MaxPool:output:0dropout_5/dropout/truediv:z:0*
T0*0
_output_shapes
:         ђї
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         ђљ
dropout_5/dropout/mul_1Muldropout_5/dropout/mul:z:0dropout_5/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђт
2separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_7_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:ђЖ
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_7_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*(
_output_shapes
:ђђ*
dtype0ѓ
)separable_conv2d_7/separable_conv2d/ShapeConst*%
valueB"      ђ      *
dtype0*
_output_shapes
:ѓ
1separable_conv2d_7/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:ѓ
-separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_5/dropout/mul_1:z:0:separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*0
_output_shapes
:         ђЄ
#separable_conv2d_7/separable_conv2dConv2D6separable_conv2d_7/separable_conv2d/depthwise:output:0<separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*0
_output_shapes
:         ђ*
T0*
strides
*
paddingVALIDК
)separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
separable_conv2d_7/BiasAddBiasAdd,separable_conv2d_7/separable_conv2d:output:01separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*0
_output_shapes
:         ђ*
T0
separable_conv2d_7/ReluRelu#separable_conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
"batch_normalization_7/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: d
"batch_normalization_7/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: й
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
dtype0┴
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ^
batch_normalization_7/ConstConst*
valueB *
dtype0*
_output_shapes
: `
batch_normalization_7/Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_7/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0$batch_normalization_7/Const:output:0&batch_normalization_7/Const_1:output:0*
T0*
U0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:b
batch_normalization_7/Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0у
9batch_normalization_7/AssignMovingAvg/Read/ReadVariableOpReadVariableOpBbatch_normalization_7_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђБ
.batch_normalization_7/AssignMovingAvg/IdentityIdentityAbatch_normalization_7/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђВ
+batch_normalization_7/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*L
_classB
@>loc:@batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp*
dtype0Ф
)batch_normalization_7/AssignMovingAvg/subSub4batch_normalization_7/AssignMovingAvg/sub/x:output:0&batch_normalization_7/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*L
_classB
@>loc:@batch_normalization_7/AssignMovingAvg/Read/ReadVariableOpъ
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOpBbatch_normalization_7_assignmovingavg_read_readvariableop_resource:^batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђК
+batch_normalization_7/AssignMovingAvg/sub_1Sub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_7/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ▓
)batch_normalization_7/AssignMovingAvg/mulMul/batch_normalization_7/AssignMovingAvg/sub_1:z:0-batch_normalization_7/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*L
_classB
@>loc:@batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђЎ
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpBbatch_normalization_7_assignmovingavg_read_readvariableop_resource-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*L
_classB
@>loc:@batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 в
;batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpDbatch_normalization_7_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђД
0batch_normalization_7/AssignMovingAvg_1/IdentityIdentityCbatch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ­
-batch_normalization_7/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*N
_classD
B@loc:@batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ▒
+batch_normalization_7/AssignMovingAvg_1/subSub6batch_normalization_7/AssignMovingAvg_1/sub/x:output:0&batch_normalization_7/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
T0ц
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOpDbatch_normalization_7_assignmovingavg_1_read_readvariableop_resource<^batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЛ
-batch_normalization_7/AssignMovingAvg_1/sub_1Sub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_7/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*N
_classD
B@loc:@batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp║
+batch_normalization_7/AssignMovingAvg_1/mulMul1batch_normalization_7/AssignMovingAvg_1/sub_1:z:0/batch_normalization_7/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*N
_classD
B@loc:@batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђБ
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpDbatch_normalization_7_assignmovingavg_1_read_readvariableop_resource/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*N
_classD
B@loc:@batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ╝
max_pooling2d_9/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*
ksize
*
paddingVALID*0
_output_shapes
:         ђ*
strides
[
dropout_6/dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: g
dropout_6/dropout/ShapeShape max_pooling2d_9/MaxPool:output:0*
_output_shapes
:*
T0i
$dropout_6/dropout/random_uniform/minConst*
_output_shapes
: *
valueB
 *    *
dtype0i
$dropout_6/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Е
.dropout_6/dropout/random_uniform/RandomUniformRandomUniform dropout_6/dropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         ђф
$dropout_6/dropout/random_uniform/subSub-dropout_6/dropout/random_uniform/max:output:0-dropout_6/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0╔
$dropout_6/dropout/random_uniform/mulMul7dropout_6/dropout/random_uniform/RandomUniform:output:0(dropout_6/dropout/random_uniform/sub:z:0*0
_output_shapes
:         ђ*
T0╗
 dropout_6/dropout/random_uniformAdd(dropout_6/dropout/random_uniform/mul:z:0-dropout_6/dropout/random_uniform/min:output:0*0
_output_shapes
:         ђ*
T0\
dropout_6/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout_6/dropout/subSub dropout_6/dropout/sub/x:output:0dropout_6/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_6/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_6/dropout/truedivRealDiv$dropout_6/dropout/truediv/x:output:0dropout_6/dropout/sub:z:0*
T0*
_output_shapes
: ░
dropout_6/dropout/GreaterEqualGreaterEqual$dropout_6/dropout/random_uniform:z:0dropout_6/dropout/rate:output:0*0
_output_shapes
:         ђ*
T0ў
dropout_6/dropout/mulMul max_pooling2d_9/MaxPool:output:0dropout_6/dropout/truediv:z:0*
T0*0
_output_shapes
:         ђї
dropout_6/dropout/CastCast"dropout_6/dropout/GreaterEqual:z:0*

DstT0*0
_output_shapes
:         ђ*

SrcT0
љ
dropout_6/dropout/mul_1Muldropout_6/dropout/mul:z:0dropout_6/dropout/Cast:y:0*
T0*0
_output_shapes
:         ђh
flatten_1/Reshape/shapeConst*
valueB"     	  *
dtype0*
_output_shapes
:ј
flatten_1/ReshapeReshapedropout_6/dropout/mul_1:z:0 flatten_1/Reshape/shape:output:0*(
_output_shapes
:         ђ*
T0┤
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђј
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0▒
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЈ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ[
dropout_7/dropout/rateConst*
_output_shapes
: *
valueB
 *џЎЎ>*
dtype0a
dropout_7/dropout/ShapeShapedense_4/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_7/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_7/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: А
.dropout_7/dropout/random_uniform/RandomUniformRandomUniform dropout_7/dropout/Shape:output:0*
dtype0*(
_output_shapes
:         ђ*
T0ф
$dropout_7/dropout/random_uniform/subSub-dropout_7/dropout/random_uniform/max:output:0-dropout_7/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ┴
$dropout_7/dropout/random_uniform/mulMul7dropout_7/dropout/random_uniform/RandomUniform:output:0(dropout_7/dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         ђ│
 dropout_7/dropout/random_uniformAdd(dropout_7/dropout/random_uniform/mul:z:0-dropout_7/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђ\
dropout_7/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout_7/dropout/subSub dropout_7/dropout/sub/x:output:0dropout_7/dropout/rate:output:0*
T0*
_output_shapes
: `
dropout_7/dropout/truediv/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?є
dropout_7/dropout/truedivRealDiv$dropout_7/dropout/truediv/x:output:0dropout_7/dropout/sub:z:0*
_output_shapes
: *
T0е
dropout_7/dropout/GreaterEqualGreaterEqual$dropout_7/dropout/random_uniform:z:0dropout_7/dropout/rate:output:0*
T0*(
_output_shapes
:         ђі
dropout_7/dropout/mulMuldense_4/Relu:activations:0dropout_7/dropout/truediv:z:0*
T0*(
_output_shapes
:         ђё
dropout_7/dropout/CastCast"dropout_7/dropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         ђ*

SrcT0
ѕ
dropout_7/dropout/mul_1Muldropout_7/dropout/mul:z:0dropout_7/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђ┤
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђЈ
dense_5/MatMulMatMuldropout_7/dropout/mul_1:z:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ▒
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЈ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:         ђ[
dropout_8/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *═╠L>a
dropout_8/dropout/ShapeShapedense_5/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_8/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_8/dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: А
.dropout_8/dropout/random_uniform/RandomUniformRandomUniform dropout_8/dropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђф
$dropout_8/dropout/random_uniform/subSub-dropout_8/dropout/random_uniform/max:output:0-dropout_8/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: ┴
$dropout_8/dropout/random_uniform/mulMul7dropout_8/dropout/random_uniform/RandomUniform:output:0(dropout_8/dropout/random_uniform/sub:z:0*(
_output_shapes
:         ђ*
T0│
 dropout_8/dropout/random_uniformAdd(dropout_8/dropout/random_uniform/mul:z:0-dropout_8/dropout/random_uniform/min:output:0*(
_output_shapes
:         ђ*
T0\
dropout_8/dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ђ
dropout_8/dropout/subSub dropout_8/dropout/sub/x:output:0dropout_8/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_8/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_8/dropout/truedivRealDiv$dropout_8/dropout/truediv/x:output:0dropout_8/dropout/sub:z:0*
T0*
_output_shapes
: е
dropout_8/dropout/GreaterEqualGreaterEqual$dropout_8/dropout/random_uniform:z:0dropout_8/dropout/rate:output:0*
T0*(
_output_shapes
:         ђі
dropout_8/dropout/mulMuldense_5/Relu:activations:0dropout_8/dropout/truediv:z:0*
T0*(
_output_shapes
:         ђё
dropout_8/dropout/CastCast"dropout_8/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:         ђѕ
dropout_8/dropout/mul_1Muldropout_8/dropout/mul:z:0dropout_8/dropout/Cast:y:0*(
_output_shapes
:         ђ*
T0│
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	ђ@ј
dense_6/MatMulMatMuldropout_8/dropout/mul_1:z:0%dense_6/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0░
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ј
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         @*
T0`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         @[
dropout_9/dropout/rateConst*
valueB
 *џЎ>*
dtype0*
_output_shapes
: a
dropout_9/dropout/ShapeShapedense_6/Relu:activations:0*
T0*
_output_shapes
:i
$dropout_9/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: i
$dropout_9/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?а
.dropout_9/dropout/random_uniform/RandomUniformRandomUniform dropout_9/dropout/Shape:output:0*'
_output_shapes
:         @*
T0*
dtype0ф
$dropout_9/dropout/random_uniform/subSub-dropout_9/dropout/random_uniform/max:output:0-dropout_9/dropout/random_uniform/min:output:0*
T0*
_output_shapes
: └
$dropout_9/dropout/random_uniform/mulMul7dropout_9/dropout/random_uniform/RandomUniform:output:0(dropout_9/dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         @▓
 dropout_9/dropout/random_uniformAdd(dropout_9/dropout/random_uniform/mul:z:0-dropout_9/dropout/random_uniform/min:output:0*'
_output_shapes
:         @*
T0\
dropout_9/dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0ђ
dropout_9/dropout/subSub dropout_9/dropout/sub/x:output:0dropout_9/dropout/rate:output:0*
_output_shapes
: *
T0`
dropout_9/dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: є
dropout_9/dropout/truedivRealDiv$dropout_9/dropout/truediv/x:output:0dropout_9/dropout/sub:z:0*
T0*
_output_shapes
: Д
dropout_9/dropout/GreaterEqualGreaterEqual$dropout_9/dropout/random_uniform:z:0dropout_9/dropout/rate:output:0*
T0*'
_output_shapes
:         @Ѕ
dropout_9/dropout/mulMuldense_6/Relu:activations:0dropout_9/dropout/truediv:z:0*'
_output_shapes
:         @*
T0Ѓ
dropout_9/dropout/CastCast"dropout_9/dropout/GreaterEqual:z:0*

SrcT0
*

DstT0*'
_output_shapes
:         @Є
dropout_9/dropout/mul_1Muldropout_9/dropout/mul:z:0dropout_9/dropout/Cast:y:0*'
_output_shapes
:         @*
T0▓
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ј
dense_7/MatMulMatMuldropout_9/dropout/mul_1:z:0%dense_7/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0░
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:ј
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         м
IdentityIdentitydense_7/Sigmoid:y:0:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp:^batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp<^batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*^separable_conv2d_6/BiasAdd/ReadVariableOp3^separable_conv2d_6/separable_conv2d/ReadVariableOp5^separable_conv2d_6/separable_conv2d/ReadVariableOp_1*^separable_conv2d_7/BiasAdd/ReadVariableOp3^separable_conv2d_7/separable_conv2d/ReadVariableOp5^separable_conv2d_7/separable_conv2d/ReadVariableOp_1*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12z
;batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_4/AssignMovingAvg_1/Read/ReadVariableOp2l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_5/AssignMovingAvg_1/Read/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_6/AssignMovingAvg_1/Read/ReadVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_12z
;batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp;batch_normalization_7/AssignMovingAvg_1/Read/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_7/AssignMovingAvg/Read/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2l
4separable_conv2d_6/separable_conv2d/ReadVariableOp_14separable_conv2d_6/separable_conv2d/ReadVariableOp_12@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2v
9batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_6/AssignMovingAvg/Read/ReadVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2h
2separable_conv2d_6/separable_conv2d/ReadVariableOp2separable_conv2d_6/separable_conv2d/ReadVariableOp2h
2separable_conv2d_7/separable_conv2d/ReadVariableOp2separable_conv2d_7/separable_conv2d/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2l
4separable_conv2d_7/separable_conv2d/ReadVariableOp_14separable_conv2d_7/separable_conv2d/ReadVariableOp_12v
9batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_5/AssignMovingAvg/Read/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12V
)separable_conv2d_7/BiasAdd/ReadVariableOp)separable_conv2d_7/BiasAdd/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2v
9batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp9batch_normalization_4/AssignMovingAvg/Read/ReadVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2V
)separable_conv2d_6/BiasAdd/ReadVariableOp)separable_conv2d_6/BiasAdd/ReadVariableOp2L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp: : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :& "
 
_user_specified_nameinputs: : : 
п	
П
D__inference_dense_4_layer_call_and_return_conditional_losses_2658885

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0* 
_output_shapes
:
ђђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
С
ђ
7__inference_batch_normalization_5_layer_call_fn_2658461

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*A
_output_shapes/
-:+                           @*
Tin	
2*.
_gradient_op_typePartitionedCall-2656052*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656051*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
░/
Ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656591

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:         @:@:@:@:@:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpЙ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpУ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: :& "
 
_user_specified_nameinputs: : : 
┘
ш
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :*
T0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         22 "
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
у
ђ
7__inference_batch_normalization_6_layer_call_fn_2658627

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*B
_output_shapes0
.:,                           ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656242*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656241*
Tout
2Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
ъ
h
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255

inputs
identityб
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4                                    {
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
А
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658652

inputs

identity_1W
IdentityIdentityinputs*0
_output_shapes
:         ђ*
T0d

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
П
ф
)__inference_dense_6_layer_call_fn_2658998

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*'
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2657084*M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078*
Tout
2*-
config_proto

GPU

CPU2*0J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
│
e
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658965

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:         ђЋ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:         ђ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*(
_output_shapes
:         ђj
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
ъ
h
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065

inputs
identityб
MaxPoolMaxPoolinputs*
paddingVALID*J
_output_shapes8
6:4                                    *
strides
*
ksize
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ъ
h
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875

inputs
identityб
MaxPoolMaxPoolinputs*
strides
*
ksize
*
paddingVALID*J
_output_shapes8
6:4                                    {
IdentityIdentityMaxPool:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
ъ
h
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
strides
*
ksize
*
paddingVALID{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
»
█
%__inference_signature_wrapper_2657568
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityѕбStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *2
Tin+
)2'*.
_gradient_op_typePartitionedCall-2657527*+
f&R$
"__inference__wrapped_model_2655652*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: :  :! :" :# :$ :% :& :. *
(
_user_specified_nameconv2d_1_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : 
Ј
ш
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2655861

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*]
_output_shapesK
I:+                            : : : : :*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*P
_input_shapes?
=:+                            ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
џ
ш
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656431

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: р
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs
▀
ф
)__inference_dense_4_layer_call_fn_2658892

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656940Ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: 
ч
e
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656880

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?Ћ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ф
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         ђЮ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: њ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:         ђ*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         ђx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         ђr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*0
_output_shapes
:         ђ*
T0b
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
ч
e
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656748

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠L>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0Ћ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*0
_output_shapes
:         ђ*
T0*
dtype0ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ф
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         ђЮ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0њ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:         ђ*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         ђx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         ђr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђb
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
│
M
1__inference_max_pooling2d_9_layer_call_fn_2656454

inputs
identity╔
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4                                    *
Tin
2*.
_gradient_op_typePartitionedCall-2656451*U
fPRN
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ѓ
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
¤	
П
D__inference_dense_7_layer_call_and_return_conditional_losses_2659044

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:         *
T0ё
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
│
M
1__inference_max_pooling2d_7_layer_call_fn_2656074

inputs
identity╔
PartitionedCallPartitionedCallinputs*J
_output_shapes8
6:4                                    *
Tin
2*.
_gradient_op_typePartitionedCall-2656071*U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Ј
┘
4__inference_separable_conv2d_4_layer_call_fn_2655725

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*-
config_proto

GPU

CPU2*0J 8*A
_output_shapes/
-:+                            *
Tin
2*.
_gradient_op_typePartitionedCall-2655719*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                            "
identityIdentity:output:0*L
_input_shapes;
9:+                           :::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : 
─
d
+__inference_dropout_8_layer_call_fn_2658975

inputs
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2657054*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657043Ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
─
d
+__inference_dropout_7_layer_call_fn_2658922

inputs
identityѕбStatefulPartitionedCall▒
StatefulPartitionedCallStatefulPartitionedCallinputs*.
_gradient_op_typePartitionedCall-2656982*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656971*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2Ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
Т/
Ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658421

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
_output_shapes
: *
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : : :& "
 
_user_specified_nameinputs: 
Ѕ
d
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658917

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:         ђ*
T0\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Ј
ш
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656051

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
ЯЇ
Ў%
 __inference__traced_save_2659301
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableopB
>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_4_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableopB
>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_5_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableopB
>savev2_separable_conv2d_6_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_6_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_6_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableopB
>savev2_separable_conv2d_7_depthwise_kernel_read_readvariableopB
>savev2_separable_conv2d_7_pointwise_kernel_read_readvariableop6
2savev2_separable_conv2d_7_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop:
6savev2_rmsprop_conv2d_1_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_1_bias_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_4_depthwise_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_4_pointwise_kernel_rms_read_readvariableopB
>savev2_rmsprop_separable_conv2d_4_bias_rms_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_4_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_4_beta_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_5_depthwise_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_5_pointwise_kernel_rms_read_readvariableopB
>savev2_rmsprop_separable_conv2d_5_bias_rms_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_5_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_5_beta_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_6_depthwise_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_6_pointwise_kernel_rms_read_readvariableopB
>savev2_rmsprop_separable_conv2d_6_bias_rms_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_6_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_6_beta_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_7_depthwise_kernel_rms_read_readvariableopN
Jsavev2_rmsprop_separable_conv2d_7_pointwise_kernel_rms_read_readvariableopB
>savev2_rmsprop_separable_conv2d_7_bias_rms_read_readvariableopF
Bsavev2_rmsprop_batch_normalization_7_gamma_rms_read_readvariableopE
Asavev2_rmsprop_batch_normalization_7_beta_rms_read_readvariableop9
5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_4_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_5_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_5_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_6_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_6_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_7_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_7_bias_rms_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_0ae599aee1384be8946682adc472c989/parts

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╚)
SaveV2/tensor_namesConst"/device:CPU:0*ы(
valueу(BС(KB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-1/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-3/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-5/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-7/pointwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-1/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-3/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-5/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/depthwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB^layer_with_weights-7/pointwise_kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBUlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Kє
SaveV2/shape_and_slicesConst"/device:CPU:0*Ф
valueАBъKB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Kо#
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop>savev2_separable_conv2d_4_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_4_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_4_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop>savev2_separable_conv2d_5_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_5_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_5_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop>savev2_separable_conv2d_6_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_6_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_6_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop>savev2_separable_conv2d_7_depthwise_kernel_read_readvariableop>savev2_separable_conv2d_7_pointwise_kernel_read_readvariableop2savev2_separable_conv2d_7_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop6savev2_rmsprop_conv2d_1_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_1_bias_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_4_depthwise_kernel_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_4_pointwise_kernel_rms_read_readvariableop>savev2_rmsprop_separable_conv2d_4_bias_rms_read_readvariableopBsavev2_rmsprop_batch_normalization_4_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_4_beta_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_5_depthwise_kernel_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_5_pointwise_kernel_rms_read_readvariableop>savev2_rmsprop_separable_conv2d_5_bias_rms_read_readvariableopBsavev2_rmsprop_batch_normalization_5_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_5_beta_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_6_depthwise_kernel_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_6_pointwise_kernel_rms_read_readvariableop>savev2_rmsprop_separable_conv2d_6_bias_rms_read_readvariableopBsavev2_rmsprop_batch_normalization_6_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_6_beta_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_7_depthwise_kernel_rms_read_readvariableopJsavev2_rmsprop_separable_conv2d_7_pointwise_kernel_rms_read_readvariableop>savev2_rmsprop_separable_conv2d_7_bias_rms_read_readvariableopBsavev2_rmsprop_batch_normalization_7_gamma_rms_read_readvariableopAsavev2_rmsprop_batch_normalization_7_beta_rms_read_readvariableop5savev2_rmsprop_dense_4_kernel_rms_read_readvariableop3savev2_rmsprop_dense_4_bias_rms_read_readvariableop5savev2_rmsprop_dense_5_kernel_rms_read_readvariableop3savev2_rmsprop_dense_5_bias_rms_read_readvariableop5savev2_rmsprop_dense_6_kernel_rms_read_readvariableop3savev2_rmsprop_dense_6_bias_rms_read_readvariableop5savev2_rmsprop_dense_7_kernel_rms_read_readvariableop3savev2_rmsprop_dense_7_bias_rms_read_readvariableop"/device:CPU:0*
_output_shapes
 *Y
dtypesO
M2K	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:ќ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0*П
_input_shapes╦
╚: :::: : : : : : : : @:@:@:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:ђ:ђ:
ђђ:ђ:
ђђ:ђ:	ђ@:@:@:: : : : : : : :::: : : : : : @:@:@:@:@:@ђ:ђ:ђ:ђ:ђ:ђђ:ђ:ђ:ђ:
ђђ:ђ:
ђђ:ђ:	ђ@:@:@:: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints:= :> :? :@ :A :B :C :D :E :F :G :H :I :J :K :L :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :' :( :) :* :+ :, :- :. :/ :0 :1 :2 :3 :4 :5 :6 :7 :8 :9 :: :; :< 
џ
ш
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658609

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ═
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: р
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
њ
┘
4__inference_separable_conv2d_7_layer_call_fn_2656295

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*X
fSRQ
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283*
Tout
2*-
config_proto

GPU

CPU2*0J 8*B
_output_shapes0
.:,                           ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656289Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
м	
П
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:	ђ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*'
_output_shapes
:         @*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
«
ђ
7__inference_batch_normalization_4_layer_call_fn_2658219

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656538*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656525*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         22 і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         22 *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs
Ѕ
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657050

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
■
b
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910

inputs
identity^
Reshape/shapeConst*
valueB"     	  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:         ђ*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
├/
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656679

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : : : :& "
 
_user_specified_nameinputs
ф
e
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657115

inputs
identityѕQ
dropout/rateConst*
valueB
 *џЎ>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*'
_output_shapes
:         @ї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*'
_output_shapes
:         @ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*'
_output_shapes
:         @R
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: Ѕ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*'
_output_shapes
:         @a
dropout/mulMulinputsdropout/truediv:z:0*
T0*'
_output_shapes
:         @o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*'
_output_shapes
:         @*

SrcT0
i
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         @Y
IdentityIdentitydropout/mul_1:z:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*&
_input_shapes
:         @:& "
 
_user_specified_nameinputs
▀
С
.__inference_sequential_1_layer_call_fn_2657459
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*2
Tin+
)2'*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-2657418*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657417*
Tout
2*-
config_proto

GPU

CPU2*0J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:. *
(
_user_specified_nameconv2d_1_input: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
«
ђ
7__inference_batch_normalization_5_layer_call_fn_2658376

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656591*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         @*
Tin	
2*.
_gradient_op_typePartitionedCall-2656616і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*>
_input_shapes-
+:         @::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs
Ј
┘
4__inference_separable_conv2d_5_layer_call_fn_2655915

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*.
_gradient_op_typePartitionedCall-2655909*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*A
_output_shapes/
-:+                           @ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*L
_input_shapes;
9:+                            :::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs
Ј
ш
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
щ/
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658587

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ч
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
ч
e
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658848

inputs
identityѕQ
dropout/rateConst*
valueB
 *═╠╠=*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
_output_shapes
:*
T0_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0Ћ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ф
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*0
_output_shapes
:         ђ*
T0Ю
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         ђR
dropout/sub/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0њ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*0
_output_shapes
:         ђj
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         ђx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         ђr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђb
IdentityIdentitydropout/mul_1:z:0*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
У
л
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283

inputs,
(separable_conv2d_readvariableop_resource.
*separable_conv2d_readvariableop_1_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбseparable_conv2d/ReadVariableOpб!separable_conv2d/ReadVariableOp_1┐
separable_conv2d/ReadVariableOpReadVariableOp(separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:ђ─
!separable_conv2d/ReadVariableOp_1ReadVariableOp*separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ђђo
separable_conv2d/ShapeConst*%
valueB"      ђ      *
dtype0*
_output_shapes
:o
separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:┘
separable_conv2d/depthwiseDepthwiseConv2dNativeinputs'separable_conv2d/ReadVariableOp:value:0*B
_output_shapes0
.:,                           ђ*
T0*
strides
*
paddingSAMEЯ
separable_conv2dConv2D#separable_conv2d/depthwise:output:0)separable_conv2d/ReadVariableOp_1:value:0*
paddingVALID*B
_output_shapes0
.:,                           ђ*
T0*
strides
А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђџ
BiasAddBiasAddseparable_conv2d:output:0BiasAdd/ReadVariableOp:value:0*B
_output_shapes0
.:,                           ђ*
T0k
ReluReluBiasAdd:output:0*
T0*B
_output_shapes0
.:,                           ђн
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^separable_conv2d/ReadVariableOp"^separable_conv2d/ReadVariableOp_1*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*M
_input_shapes<
::,                           ђ:::2B
separable_conv2d/ReadVariableOpseparable_conv2d/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2F
!separable_conv2d/ReadVariableOp_1!separable_conv2d/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : 
░/
Ќ
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656503

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *цp}?║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
: *
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpМ
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
: ─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: *
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:         22 *
T0"
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
╚ђ
Е
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657305

inputs+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_15
1separable_conv2d_4_statefulpartitionedcall_args_25
1separable_conv2d_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_45
1separable_conv2d_5_statefulpartitionedcall_args_15
1separable_conv2d_5_statefulpartitionedcall_args_25
1separable_conv2d_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_45
1separable_conv2d_6_statefulpartitionedcall_args_15
1separable_conv2d_6_statefulpartitionedcall_args_25
1separable_conv2d_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_45
1separable_conv2d_7_statefulpartitionedcall_args_15
1separable_conv2d_7_statefulpartitionedcall_args_25
1separable_conv2d_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2*
&dense_6_statefulpartitionedcall_args_1*
&dense_6_statefulpartitionedcall_args_2*
&dense_7_statefulpartitionedcall_args_1*
&dense_7_statefulpartitionedcall_args_2
identityѕб-batch_normalization_4/StatefulPartitionedCallб-batch_normalization_5/StatefulPartitionedCallб-batch_normalization_6/StatefulPartitionedCallб-batch_normalization_7/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallбdense_4/StatefulPartitionedCallбdense_5/StatefulPartitionedCallбdense_6/StatefulPartitionedCallбdense_7/StatefulPartitionedCallб!dropout_5/StatefulPartitionedCallб!dropout_6/StatefulPartitionedCallб!dropout_7/StatefulPartitionedCallб!dropout_8/StatefulPartitionedCallб!dropout_9/StatefulPartitionedCallб*separable_conv2d_4/StatefulPartitionedCallб*separable_conv2d_5/StatefulPartitionedCallб*separable_conv2d_6/StatefulPartitionedCallб*separable_conv2d_7/StatefulPartitionedCallћ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputs'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666*
Tout
2*-
config_proto

GPU

CPU2*0J 8*/
_output_shapes
:         dd*
Tin
2*.
_gradient_op_typePartitionedCall-2655672р
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*U
fPRN
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         22*.
_gradient_op_typePartitionedCall-2655691њ
*separable_conv2d_4/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:01separable_conv2d_4_statefulpartitionedcall_args_11separable_conv2d_4_statefulpartitionedcall_args_21separable_conv2d_4_statefulpartitionedcall_args_3*/
_output_shapes
:         22 *
Tin
2*.
_gradient_op_typePartitionedCall-2655719*X
fSRQ
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713*
Tout
2*-
config_proto

GPU

CPU2*0J 8с
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_4/StatefulPartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*[
fVRT
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656503*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         22 *.
_gradient_op_typePartitionedCall-2656528Ь
max_pooling2d_6/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2655881*U
fPRN
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:          њ
*separable_conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_6/PartitionedCall:output:01separable_conv2d_5_statefulpartitionedcall_args_11separable_conv2d_5_statefulpartitionedcall_args_21separable_conv2d_5_statefulpartitionedcall_args_3*
Tin
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2655909*X
fSRQ
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903*
Tout
2*-
config_proto

GPU

CPU2*0J 8с
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_5/StatefulPartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*.
_gradient_op_typePartitionedCall-2656616*[
fVRT
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656591*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*/
_output_shapes
:         @Ь
max_pooling2d_7/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*/
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2656071*U
fPRN
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065Њ
*separable_conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_7/PartitionedCall:output:01separable_conv2d_6_statefulpartitionedcall_args_11separable_conv2d_6_statefulpartitionedcall_args_21separable_conv2d_6_statefulpartitionedcall_args_3*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656099*X
fSRQ
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093*
Tout
2*-
config_proto

GPU

CPU2*0J 8С
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_6/StatefulPartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656679*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656704№
max_pooling2d_8/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656261*U
fPRN
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255*
Tout
2т
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_8/PartitionedCall:output:0*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656759*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656748*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ћ
*separable_conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:01separable_conv2d_7_statefulpartitionedcall_args_11separable_conv2d_7_statefulpartitionedcall_args_21separable_conv2d_7_statefulpartitionedcall_args_3*X
fSRQ
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656289С
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3separable_conv2d_7/StatefulPartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656836*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656811*
Tout
2№
max_pooling2d_9/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656451*U
fPRN
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђЅ
!dropout_6/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_9/PartitionedCall:output:0"^dropout_5/StatefulPartitionedCall*.
_gradient_op_typePartitionedCall-2656891*O
fJRH
F__inference_dropout_6_layer_call_and_return_conditional_losses_2656880*
Tout
2*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin
2¤
flatten_1/PartitionedCallPartitionedCall*dropout_6/StatefulPartitionedCall:output:0*.
_gradient_op_typePartitionedCall-2656916*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђЦ
dense_4/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2656940*M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2656934*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђЂ
!dropout_7/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0"^dropout_6/StatefulPartitionedCall*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2656982*O
fJRH
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656971*
Tout
2Г
dense_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_7/StatefulPartitionedCall:output:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*M
fHRF
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006*
Tout
2*-
config_proto

GPU

CPU2*0J 8*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2657012Ђ
!dropout_8/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0"^dropout_7/StatefulPartitionedCall*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2657054*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657043г
dense_6/StatefulPartitionedCallStatefulPartitionedCall*dropout_8/StatefulPartitionedCall:output:0&dense_6_statefulpartitionedcall_args_1&dense_6_statefulpartitionedcall_args_2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         @*.
_gradient_op_typePartitionedCall-2657084*M
fHRF
D__inference_dense_6_layer_call_and_return_conditional_losses_2657078*
Tout
2ђ
!dropout_9/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0"^dropout_8/StatefulPartitionedCall*O
fJRH
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657115*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         @*
Tin
2*.
_gradient_op_typePartitionedCall-2657126г
dense_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_9/StatefulPartitionedCall:output:0&dense_7_statefulpartitionedcall_args_1&dense_7_statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657156*M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150*
Tout
2*-
config_proto

GPU

CPU2*0J 8*
Tin
2*'
_output_shapes
:         ├
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall"^dropout_6/StatefulPartitionedCall"^dropout_7/StatefulPartitionedCall"^dropout_8/StatefulPartitionedCall"^dropout_9/StatefulPartitionedCall+^separable_conv2d_4/StatefulPartitionedCall+^separable_conv2d_5/StatefulPartitionedCall+^separable_conv2d_6/StatefulPartitionedCall+^separable_conv2d_7/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2F
!dropout_6/StatefulPartitionedCall!dropout_6/StatefulPartitionedCall2F
!dropout_7/StatefulPartitionedCall!dropout_7/StatefulPartitionedCall2F
!dropout_8/StatefulPartitionedCall!dropout_8/StatefulPartitionedCall2X
*separable_conv2d_4/StatefulPartitionedCall*separable_conv2d_4/StatefulPartitionedCall2F
!dropout_9/StatefulPartitionedCall!dropout_9/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2X
*separable_conv2d_5/StatefulPartitionedCall*separable_conv2d_5/StatefulPartitionedCall2X
*separable_conv2d_6/StatefulPartitionedCall*separable_conv2d_6/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2X
*separable_conv2d_7/StatefulPartitionedCall*separable_conv2d_7/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
▒
ђ
7__inference_batch_normalization_7_layer_call_fn_2658752

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*0
_output_shapes
:         ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656846*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656833*
Tout
2*-
config_proto

GPU

CPU2*0J 8І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
▄
ф
)__inference_dense_7_layer_call_fn_2659051

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*.
_gradient_op_typePartitionedCall-2657156*M
fHRF
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150*
Tout
2*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *
Tin
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
│
e
F__inference_dropout_7_layer_call_and_return_conditional_losses_2656971

inputs
identityѕQ
dropout/rateConst*
valueB
 *џЎЎ>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Б
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*(
_output_shapes
:         ђ*
T0Ћ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:         ђR
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
_output_shapes
: *
valueB
 *  ђ?*
dtype0h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: і
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*
T0*(
_output_shapes
:         ђb
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:         ђ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
Т/
Ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656017

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: :& "
 
_user_specified_nameinputs: : : 
є
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_2657122

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:         @[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:& "
 
_user_specified_nameinputs
у
ђ
7__inference_batch_normalization_6_layer_call_fn_2658618

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*
Tin	
2*B
_output_shapes0
.:,                           ђ*.
_gradient_op_typePartitionedCall-2656208*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656207*
Tout
2Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
ё
я
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:Ф
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingSAME*A
_output_shapes/
-:+                           *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+                           *
T0j
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+                           Ц
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           "
identityIdentity:output:0*H
_input_shapes7
5:+                           ::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
А
d
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656755

inputs

identity_1W
IdentityIdentityinputs*
T0*0
_output_shapes
:         ђd

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
░/
Ќ
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658345

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         @:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
┘
ш
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2656525

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         22 "
identityIdentity:output:0*>
_input_shapes-
+:         22 ::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
┘
ш
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2656613

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         @:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         @"
identityIdentity:output:0*>
_input_shapes-
+:         @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
▀
С
.__inference_sequential_1_layer_call_fn_2657347
conv2d_1_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallconv2d_1_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*-
config_proto

GPU

CPU2*0J 8*2
Tin+
)2'*'
_output_shapes
:         *.
_gradient_op_typePartitionedCall-2657306*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657305*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :. *
(
_user_specified_nameconv2d_1_input: : : 
▒
ђ
7__inference_batch_normalization_6_layer_call_fn_2658551

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*-
config_proto

GPU

CPU2*0J 8*0
_output_shapes
:         ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656714*[
fVRT
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2656701*
Tout
2І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
▄
d
+__inference_dropout_5_layer_call_fn_2658657

inputs
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*0
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656759*O
fJRH
F__inference_dropout_5_layer_call_and_return_conditional_losses_2656748*
Tout
2І
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
└
G
+__inference_dropout_8_layer_call_fn_2658980

inputs
identityА
PartitionedCallPartitionedCallinputs*(
_output_shapes
:         ђ*
Tin
2*.
_gradient_op_typePartitionedCall-2657062*O
fJRH
F__inference_dropout_8_layer_call_and_return_conditional_losses_2657050*
Tout
2*-
config_proto

GPU

CPU2*0J 8a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
є
d
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659023

inputs

identity_1N
IdentityIdentityinputs*'
_output_shapes
:         @*
T0[

Identity_1IdentityIdentity:output:0*'
_output_shapes
:         @*
T0"!

identity_1Identity_1:output:0*&
_input_shapes
:         @:& "
 
_user_specified_nameinputs
м	
П
D__inference_dense_6_layer_call_and_return_conditional_losses_2658991

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	ђ@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @P
ReluReluBiasAdd:output:0*'
_output_shapes
:         @*
T0І
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*/
_input_shapes
:         ђ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : :& "
 
_user_specified_nameinputs
├/
Ќ
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658511

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
С
ш
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658533

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ│
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђи
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ╗
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ¤
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
Ѕ
d
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658970

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*'
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
у
ђ
7__inference_batch_normalization_7_layer_call_fn_2658819

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall█
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*B
_output_shapes0
.:,                           ђ*
Tin	
2*.
_gradient_op_typePartitionedCall-2656398*[
fVRT
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656397*
Tout
2*-
config_proto

GPU

CPU2*0J 8Ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,                           ђ"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
ч
e
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658647

inputs
identityѕQ
dropout/rateConst*
_output_shapes
: *
valueB
 *═╠L>*
dtype0C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: Ћ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*0
_output_shapes
:         ђї
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Ф
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*0
_output_shapes
:         ђЮ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*0
_output_shapes
:         ђR
dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  ђ?b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ђ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: њ
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*0
_output_shapes
:         ђ*
T0j
dropout/mulMulinputsdropout/truediv:z:0*
T0*0
_output_shapes
:         ђx
dropout/CastCastdropout/GreaterEqual:z:0*

SrcT0
*

DstT0*0
_output_shapes
:         ђr
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:         ђb
IdentityIdentitydropout/mul_1:z:0*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
┼Т
ј
I__inference_sequential_1_layer_call_and_return_conditional_losses_2658043

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource?
;separable_conv2d_4_separable_conv2d_readvariableop_resourceA
=separable_conv2d_4_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_4_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource?
;separable_conv2d_5_separable_conv2d_readvariableop_resourceA
=separable_conv2d_5_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_5_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource?
;separable_conv2d_6_separable_conv2d_readvariableop_resourceA
=separable_conv2d_6_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_6_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource?
;separable_conv2d_7_separable_conv2d_readvariableop_resourceA
=separable_conv2d_7_separable_conv2d_readvariableop_1_resource6
2separable_conv2d_7_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource
identityѕб5batch_normalization_4/FusedBatchNormV3/ReadVariableOpб7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_4/ReadVariableOpб&batch_normalization_4/ReadVariableOp_1б5batch_normalization_5/FusedBatchNormV3/ReadVariableOpб7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_5/ReadVariableOpб&batch_normalization_5/ReadVariableOp_1б5batch_normalization_6/FusedBatchNormV3/ReadVariableOpб7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_6/ReadVariableOpб&batch_normalization_6/ReadVariableOp_1б5batch_normalization_7/FusedBatchNormV3/ReadVariableOpб7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_7/ReadVariableOpб&batch_normalization_7/ReadVariableOp_1бconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpбdense_4/BiasAdd/ReadVariableOpбdense_4/MatMul/ReadVariableOpбdense_5/BiasAdd/ReadVariableOpбdense_5/MatMul/ReadVariableOpбdense_6/BiasAdd/ReadVariableOpбdense_6/MatMul/ReadVariableOpбdense_7/BiasAdd/ReadVariableOpбdense_7/MatMul/ReadVariableOpб)separable_conv2d_4/BiasAdd/ReadVariableOpб2separable_conv2d_4/separable_conv2d/ReadVariableOpб4separable_conv2d_4/separable_conv2d/ReadVariableOp_1б)separable_conv2d_5/BiasAdd/ReadVariableOpб2separable_conv2d_5/separable_conv2d/ReadVariableOpб4separable_conv2d_5/separable_conv2d/ReadVariableOp_1б)separable_conv2d_6/BiasAdd/ReadVariableOpб2separable_conv2d_6/separable_conv2d/ReadVariableOpб4separable_conv2d_6/separable_conv2d/ReadVariableOp_1б)separable_conv2d_7/BiasAdd/ReadVariableOpб2separable_conv2d_7/separable_conv2d/ReadVariableOpб4separable_conv2d_7/separable_conv2d/ReadVariableOp_1╝
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:Ф
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         dd*
T0*
strides
*
paddingSAME▓
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         ddj
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:         ddг
max_pooling2d_5/MaxPoolMaxPoolconv2d_1/Relu:activations:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:         22С
2separable_conv2d_4/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_4_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:У
4separable_conv2d_4/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_4_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: ѓ
)separable_conv2d_4/separable_conv2d/ShapeConst*
_output_shapes
:*%
valueB"            *
dtype0ѓ
1separable_conv2d_4/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:є
-separable_conv2d_4/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_5/MaxPool:output:0:separable_conv2d_4/separable_conv2d/ReadVariableOp:value:0*/
_output_shapes
:         22*
T0*
strides
*
paddingSAMEє
#separable_conv2d_4/separable_conv2dConv2D6separable_conv2d_4/separable_conv2d/depthwise:output:0<separable_conv2d_4/separable_conv2d/ReadVariableOp_1:value:0*
paddingVALID*/
_output_shapes
:         22 *
T0*
strides
к
)separable_conv2d_4/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0└
separable_conv2d_4/BiasAddBiasAdd,separable_conv2d_4/separable_conv2d:output:01separable_conv2d_4/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         22 *
T0~
separable_conv2d_4/ReluRelu#separable_conv2d_4/BiasAdd:output:0*/
_output_shapes
:         22 *
T0d
"batch_normalization_4/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: d
"batch_normalization_4/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: ╝
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
dtype0└
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: я
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: Р
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: ├
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         22 : : : : :*
T0`
batch_normalization_4/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
max_pooling2d_6/MaxPoolMaxPool*batch_normalization_4/FusedBatchNormV3:y:0*
strides
*
ksize
*
paddingVALID*/
_output_shapes
:          С
2separable_conv2d_5/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_5_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
: *
dtype0У
4separable_conv2d_5/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_5_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
: @ѓ
)separable_conv2d_5/separable_conv2d/ShapeConst*%
valueB"             *
dtype0*
_output_shapes
:ѓ
1separable_conv2d_5/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:є
-separable_conv2d_5/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_6/MaxPool:output:0:separable_conv2d_5/separable_conv2d/ReadVariableOp:value:0*
T0*
strides
*
paddingSAME*/
_output_shapes
:          є
#separable_conv2d_5/separable_conv2dConv2D6separable_conv2d_5/separable_conv2d/depthwise:output:0<separable_conv2d_5/separable_conv2d/ReadVariableOp_1:value:0*
paddingVALID*/
_output_shapes
:         @*
T0*
strides
к
)separable_conv2d_5/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0└
separable_conv2d_5/BiasAddBiasAdd,separable_conv2d_5/separable_conv2d:output:01separable_conv2d_5/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         @*
T0~
separable_conv2d_5/ReluRelu#separable_conv2d_5/BiasAdd:output:0*/
_output_shapes
:         @*
T0d
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
d
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
а
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: ╝
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@└
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@я
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Р
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@├
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         @:@:@:@:@:`
batch_normalization_5/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
max_pooling2d_7/MaxPoolMaxPool*batch_normalization_5/FusedBatchNormV3:y:0*/
_output_shapes
:         @*
strides
*
ksize
*
paddingVALIDС
2separable_conv2d_6/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_6_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@ж
4separable_conv2d_6/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_6_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:@ђѓ
)separable_conv2d_6/separable_conv2d/ShapeConst*%
valueB"      @      *
dtype0*
_output_shapes
:ѓ
1separable_conv2d_6/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:є
-separable_conv2d_6/separable_conv2d/depthwiseDepthwiseConv2dNative max_pooling2d_7/MaxPool:output:0:separable_conv2d_6/separable_conv2d/ReadVariableOp:value:0*
paddingSAME*/
_output_shapes
:         @*
T0*
strides
Є
#separable_conv2d_6/separable_conv2dConv2D6separable_conv2d_6/separable_conv2d/depthwise:output:0<separable_conv2d_6/separable_conv2d/ReadVariableOp_1:value:0*0
_output_shapes
:         ђ*
T0*
strides
*
paddingVALIDК
)separable_conv2d_6/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
separable_conv2d_6/BiasAddBiasAdd,separable_conv2d_6/separable_conv2d:output:01separable_conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
separable_conv2d_6/ReluRelu#separable_conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:         ђd
"batch_normalization_6/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: d
"batch_normalization_6/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: й
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ▀
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђс
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ╚
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:`
batch_normalization_6/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╝
max_pooling2d_8/MaxPoolMaxPool*batch_normalization_6/FusedBatchNormV3:y:0*
strides
*
ksize
*
paddingVALID*0
_output_shapes
:         ђ{
dropout_5/IdentityIdentity max_pooling2d_8/MaxPool:output:0*
T0*0
_output_shapes
:         ђт
2separable_conv2d_7/separable_conv2d/ReadVariableOpReadVariableOp;separable_conv2d_7_separable_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*'
_output_shapes
:ђЖ
4separable_conv2d_7/separable_conv2d/ReadVariableOp_1ReadVariableOp=separable_conv2d_7_separable_conv2d_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*(
_output_shapes
:ђђѓ
)separable_conv2d_7/separable_conv2d/ShapeConst*%
valueB"      ђ      *
dtype0*
_output_shapes
:ѓ
1separable_conv2d_7/separable_conv2d/dilation_rateConst*
valueB"      *
dtype0*
_output_shapes
:ѓ
-separable_conv2d_7/separable_conv2d/depthwiseDepthwiseConv2dNativedropout_5/Identity:output:0:separable_conv2d_7/separable_conv2d/ReadVariableOp:value:0*
strides
*
paddingSAME*0
_output_shapes
:         ђ*
T0Є
#separable_conv2d_7/separable_conv2dConv2D6separable_conv2d_7/separable_conv2d/depthwise:output:0<separable_conv2d_7/separable_conv2d/ReadVariableOp_1:value:0*
T0*
strides
*
paddingVALID*0
_output_shapes
:         ђК
)separable_conv2d_7/BiasAdd/ReadVariableOpReadVariableOp2separable_conv2d_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
separable_conv2d_7/BiasAddBiasAdd,separable_conv2d_7/separable_conv2d:output:01separable_conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:         ђ
separable_conv2d_7/ReluRelu#separable_conv2d_7/BiasAdd:output:0*0
_output_shapes
:         ђ*
T0d
"batch_normalization_7/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: d
"batch_normalization_7/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: а
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: й
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ┴
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ▀
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђс
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ╚
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%separable_conv2d_7/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:`
batch_normalization_7/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╝
max_pooling2d_9/MaxPoolMaxPool*batch_normalization_7/FusedBatchNormV3:y:0*
ksize
*
paddingVALID*0
_output_shapes
:         ђ*
strides
{
dropout_6/IdentityIdentity max_pooling2d_9/MaxPool:output:0*
T0*0
_output_shapes
:         ђh
flatten_1/Reshape/shapeConst*
valueB"     	  *
dtype0*
_output_shapes
:ј
flatten_1/ReshapeReshapedropout_6/Identity:output:0 flatten_1/Reshape/shape:output:0*(
_output_shapes
:         ђ*
T0┤
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
ђђ*
dtype0ј
dense_4/MatMulMatMulflatten_1/Reshape:output:0%dense_4/MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0▒
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЈ
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:         ђm
dropout_7/IdentityIdentitydense_4/Relu:activations:0*
T0*(
_output_shapes
:         ђ┤
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
ђђ*
dtype0Ј
dense_5/MatMulMatMuldropout_7/Identity:output:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ▒
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЈ
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђa
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:         ђm
dropout_8/IdentityIdentitydense_5/Relu:activations:0*
T0*(
_output_shapes
:         ђ│
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	ђ@ј
dense_6/MatMulMatMuldropout_8/Identity:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @░
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ј
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @`
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:         @l
dropout_9/IdentityIdentitydense_6/Relu:activations:0*'
_output_shapes
:         @*
T0▓
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@ј
dense_7/MatMulMatMuldropout_9/Identity:output:0%dense_7/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0░
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:ј
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_7/SigmoidSigmoiddense_7/BiasAdd:output:0*
T0*'
_output_shapes
:         і
IdentityIdentitydense_7/Sigmoid:y:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*^separable_conv2d_4/BiasAdd/ReadVariableOp3^separable_conv2d_4/separable_conv2d/ReadVariableOp5^separable_conv2d_4/separable_conv2d/ReadVariableOp_1*^separable_conv2d_5/BiasAdd/ReadVariableOp3^separable_conv2d_5/separable_conv2d/ReadVariableOp5^separable_conv2d_5/separable_conv2d/ReadVariableOp_1*^separable_conv2d_6/BiasAdd/ReadVariableOp3^separable_conv2d_6/separable_conv2d/ReadVariableOp5^separable_conv2d_6/separable_conv2d/ReadVariableOp_1*^separable_conv2d_7/BiasAdd/ReadVariableOp3^separable_conv2d_7/separable_conv2d/ReadVariableOp5^separable_conv2d_7/separable_conv2d/ReadVariableOp_1*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12l
4separable_conv2d_5/separable_conv2d/ReadVariableOp_14separable_conv2d_5/separable_conv2d/ReadVariableOp_12V
)separable_conv2d_6/BiasAdd/ReadVariableOp)separable_conv2d_6/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2l
4separable_conv2d_6/separable_conv2d/ReadVariableOp_14separable_conv2d_6/separable_conv2d/ReadVariableOp_12>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2V
)separable_conv2d_5/BiasAdd/ReadVariableOp)separable_conv2d_5/BiasAdd/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12l
4separable_conv2d_4/separable_conv2d/ReadVariableOp_14separable_conv2d_4/separable_conv2d/ReadVariableOp_12P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2h
2separable_conv2d_4/separable_conv2d/ReadVariableOp2separable_conv2d_4/separable_conv2d/ReadVariableOp2n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2h
2separable_conv2d_5/separable_conv2d/ReadVariableOp2separable_conv2d_5/separable_conv2d/ReadVariableOp2V
)separable_conv2d_4/BiasAdd/ReadVariableOp)separable_conv2d_4/BiasAdd/ReadVariableOp2h
2separable_conv2d_6/separable_conv2d/ReadVariableOp2separable_conv2d_6/separable_conv2d/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12h
2separable_conv2d_7/separable_conv2d/ReadVariableOp2separable_conv2d_7/separable_conv2d/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2l
4separable_conv2d_7/separable_conv2d/ReadVariableOp_14separable_conv2d_7/separable_conv2d/ReadVariableOp_12P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12V
)separable_conv2d_7/BiasAdd/ReadVariableOp)separable_conv2d_7/BiasAdd/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1: : : : : : : : : : : : :  :! :" :# :$ :% :& :& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : 
├/
Ќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2656811

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ш
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*P
_output_shapes>
<:         ђ:ђ:ђ:ђ:ђ:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes	
:ђ*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђ*
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ж
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*0
_output_shapes
:         ђ*
T0"
identityIdentity:output:0*?
_input_shapes.
,:         ђ::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
К
▄
.__inference_sequential_1_layer_call_fn_2658129

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*-
config_proto

GPU

CPU2*0J 8*'
_output_shapes
:         *2
Tin+
)2'*.
_gradient_op_typePartitionedCall-2657418*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657417*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
¤	
П
D__inference_dense_7_layer_call_and_return_conditional_losses_2657150

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*'
_output_shapes
:         *
T0ё
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
щ/
Ќ
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658788

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: Љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђЋ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђH
ConstConst*
_output_shapes
: *
valueB *
dtype0J
Const_1Const*
_output_shapes
: *
valueB *
dtype0ѕ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*b
_output_shapesP
N:,                           ђ:ђ:ђ:ђ:ђ:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╗
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: ▄
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ№
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђ┌
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes	
:ђФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 ┐
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђ{
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes	
:ђ─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: Р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђщ
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes	
:ђ*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpР
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes	
:ђ*
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 ч
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*B
_output_shapes0
.:,                           ђ*
T0"
identityIdentity:output:0*Q
_input_shapes@
>:,                           ђ::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
К
▄
.__inference_sequential_1_layer_call_fn_2658086

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38*.
_gradient_op_typePartitionedCall-2657306*R
fMRK
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657305*
Tout
2*-
config_proto

GPU

CPU2*0J 8*2
Tin+
)2'*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*╚
_input_shapesХ
│:         dd::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :& "
 
_user_specified_nameinputs: : : : : : 
╚
G
+__inference_flatten_1_layer_call_fn_2658874

inputs
identityА
PartitionedCallPartitionedCallinputs*-
config_proto

GPU

CPU2*0J 8*
Tin
2*(
_output_shapes
:         ђ*.
_gradient_op_typePartitionedCall-2656916*O
fJRH
F__inference_flatten_1_layer_call_and_return_conditional_losses_2656910*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ:& "
 
_user_specified_nameinputs
п	
П
D__inference_dense_5_layer_call_and_return_conditional_losses_2657006

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpц
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0* 
_output_shapes
:
ђђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes	
:ђw
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:         ђ*
T0ї
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*/
_input_shapes
:         ђ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*└
serving_defaultг
Q
conv2d_1_input?
 serving_default_conv2d_1_input:0         dd;
dense_70
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:«Щ
ве
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
layer-11
layer-12
layer_with_weights-7
layer-13
layer_with_weights-8
layer-14
layer-15
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer_with_weights-10
layer-20
layer-21
layer_with_weights-11
layer-22
layer-23
layer_with_weights-12
layer-24
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api

signatures
┼__call__
+к&call_and_return_all_conditional_losses
К_default_save_signature"зА
_tf_keras_sequentialМА{"class_name": "Sequential", "name": "sequential_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_1", "layers": [{"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "SeparableConv2D", "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Dropout", "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 2.429999995001708e-06, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
┼
 	variables
!trainable_variables
"regularization_losses
#	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses"┤
_tf_keras_layerџ{"class_name": "InputLayer", "name": "conv2d_1_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 100, 100, 3], "config": {"batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "sparse": false, "name": "conv2d_1_input"}}
д

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
╩__call__
+╦&call_and_return_all_conditional_losses" 
_tf_keras_layerт{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 100, 100, 3], "config": {"name": "conv2d_1", "trainable": true, "batch_input_shape": [null, 100, 100, 3], "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
 
*	variables
+trainable_variables
,regularization_losses
-	keras_api
╠__call__
+═&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
С

.depthwise_kernel
/pointwise_kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
╬__call__
+¤&call_and_return_all_conditional_losses"Ю	
_tf_keras_layerЃ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_4", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}}}
х
5axis
	6gamma
7beta
8moving_mean
9moving_variance
:	variables
;trainable_variables
<regularization_losses
=	keras_api
л__call__
+Л&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
 
>	variables
?trainable_variables
@regularization_losses
A	keras_api
м__call__
+М&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_6", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
С

Bdepthwise_kernel
Cpointwise_kernel
Dbias
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
н__call__
+Н&call_and_return_all_conditional_losses"Ю	
_tf_keras_layerЃ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
х
Iaxis
	Jgamma
Kbeta
Lmoving_mean
Mmoving_variance
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
о__call__
+О&call_and_return_all_conditional_losses"▀
_tf_keras_layer┼{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
 
R	variables
Strainable_variables
Tregularization_losses
U	keras_api
п__call__
+┘&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_7", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
т

Vdepthwise_kernel
Wpointwise_kernel
Xbias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
┌__call__
+█&call_and_return_all_conditional_losses"ъ	
_tf_keras_layerё	{"class_name": "SeparableConv2D", "name": "separable_conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
Х
]axis
	^gamma
_beta
`moving_mean
amoving_variance
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
▄__call__
+П&call_and_return_all_conditional_losses"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
 
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
я__call__
+▀&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_8", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
▒
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
Я__call__
+р&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Dropout", "name": "dropout_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_5", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Т

ndepthwise_kernel
opointwise_kernel
pbias
q	variables
rtrainable_variables
sregularization_losses
t	keras_api
Р__call__
+с&call_and_return_all_conditional_losses"Ъ	
_tf_keras_layerЁ	{"class_name": "SeparableConv2D", "name": "separable_conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "separable_conv2d_7", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "pointwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "pointwise_regularizer": null, "depthwise_constraint": null, "pointwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
Х
uaxis
	vgamma
wbeta
xmoving_mean
ymoving_variance
z	variables
{trainable_variables
|regularization_losses
}	keras_api
С__call__
+т&call_and_return_all_conditional_losses"Я
_tf_keras_layerк{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 256}}}}
Ђ
~	variables
trainable_variables
ђregularization_losses
Ђ	keras_api
Т__call__
+у&call_and_return_all_conditional_losses"Ь
_tf_keras_layerн{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
х
ѓ	variables
Ѓtrainable_variables
ёregularization_losses
Ё	keras_api
У__call__
+ж&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Dropout", "name": "dropout_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_6", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Х
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
Ж__call__
+в&call_and_return_all_conditional_losses"А
_tf_keras_layerЄ{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ч
іkernel
	Іbias
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
В__call__
+ь&call_and_return_all_conditional_losses"¤
_tf_keras_layerх{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}}
х
љ	variables
Љtrainable_variables
њregularization_losses
Њ	keras_api
Ь__call__
+№&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Dropout", "name": "dropout_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_7", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}
ч
ћkernel
	Ћbias
ќ	variables
Ќtrainable_variables
ўregularization_losses
Ў	keras_api
­__call__
+ы&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}}
х
џ	variables
Џtrainable_variables
юregularization_losses
Ю	keras_api
Ы__call__
+з&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"class_name": "Dropout", "name": "dropout_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_8", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Щ
ъkernel
	Ъbias
а	variables
Аtrainable_variables
бregularization_losses
Б	keras_api
З__call__
+ш&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
Х
ц	variables
Цtrainable_variables
дregularization_losses
Д	keras_api
Ш__call__
+э&call_and_return_all_conditional_losses"А
_tf_keras_layerЄ{"class_name": "Dropout", "name": "dropout_9", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout_9", "trainable": true, "dtype": "float32", "rate": 0.15, "noise_shape": null, "seed": null}}
ч
еkernel
	Еbias
ф	variables
Фtrainable_variables
гregularization_losses
Г	keras_api
Э__call__
+щ&call_and_return_all_conditional_losses"╬
_tf_keras_layer┤{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}
К
	«iter

»decay
░learning_rate
▒momentum
▓rho
$rmsД
%rmsе
.rmsЕ
/rmsф
0rmsФ
6rmsг
7rmsГ
Brms«
Crms»
Drms░
Jrms▒
Krms▓
Vrms│
Wrms┤
Xrmsх
^rmsХ
_rmsи
nrmsИ
orms╣
prms║
vrms╗
wrms╝іrmsйІrmsЙћrms┐Ћrms└ъrms┴Ъrms┬еrms├Еrms─"
	optimizer
╬
$0
%1
.2
/3
04
65
76
87
98
B9
C10
D11
J12
K13
L14
M15
V16
W17
X18
^19
_20
`21
a22
n23
o24
p25
v26
w27
x28
y29
і30
І31
ћ32
Ћ33
ъ34
Ъ35
е36
Е37"
trackable_list_wrapper
ј
$0
%1
.2
/3
04
65
76
B7
C8
D9
J10
K11
V12
W13
X14
^15
_16
n17
o18
p19
v20
w21
і22
І23
ћ24
Ћ25
ъ26
Ъ27
е28
Е29"
trackable_list_wrapper
 "
trackable_list_wrapper
┐
│layers
	variables
trainable_variables
regularization_losses
 ┤layer_regularization_losses
хmetrics
Хnon_trainable_variables
┼__call__
К_default_save_signature
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
-
Щserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
иlayers
 	variables
!trainable_variables
"regularization_losses
 Иlayer_regularization_losses
╣metrics
║non_trainable_variables
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_1/kernel
:2conv2d_1/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
╗layers
&	variables
'trainable_variables
(regularization_losses
 ╝layer_regularization_losses
йmetrics
Йnon_trainable_variables
╩__call__
+╦&call_and_return_all_conditional_losses
'╦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
┐layers
*	variables
+trainable_variables
,regularization_losses
 └layer_regularization_losses
┴metrics
┬non_trainable_variables
╠__call__
+═&call_and_return_all_conditional_losses
'═"call_and_return_conditional_losses"
_generic_user_object
=:;2#separable_conv2d_4/depthwise_kernel
=:; 2#separable_conv2d_4/pointwise_kernel
%:# 2separable_conv2d_4/bias
5
.0
/1
02"
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
А
├layers
1	variables
2trainable_variables
3regularization_losses
 ─layer_regularization_losses
┼metrics
кnon_trainable_variables
╬__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_4/gamma
(:& 2batch_normalization_4/beta
1:/  (2!batch_normalization_4/moving_mean
5:3  (2%batch_normalization_4/moving_variance
<
60
71
82
93"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
А
Кlayers
:	variables
;trainable_variables
<regularization_losses
 ╚layer_regularization_losses
╔metrics
╩non_trainable_variables
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
╦layers
>	variables
?trainable_variables
@regularization_losses
 ╠layer_regularization_losses
═metrics
╬non_trainable_variables
м__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
=:; 2#separable_conv2d_5/depthwise_kernel
=:; @2#separable_conv2d_5/pointwise_kernel
%:#@2separable_conv2d_5/bias
5
B0
C1
D2"
trackable_list_wrapper
5
B0
C1
D2"
trackable_list_wrapper
 "
trackable_list_wrapper
А
¤layers
E	variables
Ftrainable_variables
Gregularization_losses
 лlayer_regularization_losses
Лmetrics
мnon_trainable_variables
н__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
<
J0
K1
L2
M3"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
Мlayers
N	variables
Otrainable_variables
Pregularization_losses
 нlayer_regularization_losses
Нmetrics
оnon_trainable_variables
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Оlayers
R	variables
Strainable_variables
Tregularization_losses
 пlayer_regularization_losses
┘metrics
┌non_trainable_variables
п__call__
+┘&call_and_return_all_conditional_losses
'┘"call_and_return_conditional_losses"
_generic_user_object
=:;@2#separable_conv2d_6/depthwise_kernel
>:<@ђ2#separable_conv2d_6/pointwise_kernel
&:$ђ2separable_conv2d_6/bias
5
V0
W1
X2"
trackable_list_wrapper
5
V0
W1
X2"
trackable_list_wrapper
 "
trackable_list_wrapper
А
█layers
Y	variables
Ztrainable_variables
[regularization_losses
 ▄layer_regularization_losses
Пmetrics
яnon_trainable_variables
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_6/gamma
):'ђ2batch_normalization_6/beta
2:0ђ (2!batch_normalization_6/moving_mean
6:4ђ (2%batch_normalization_6/moving_variance
<
^0
_1
`2
a3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
▀layers
b	variables
ctrainable_variables
dregularization_losses
 Яlayer_regularization_losses
рmetrics
Рnon_trainable_variables
▄__call__
+П&call_and_return_all_conditional_losses
'П"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
сlayers
f	variables
gtrainable_variables
hregularization_losses
 Сlayer_regularization_losses
тmetrics
Тnon_trainable_variables
я__call__
+▀&call_and_return_all_conditional_losses
'▀"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
уlayers
j	variables
ktrainable_variables
lregularization_losses
 Уlayer_regularization_losses
жmetrics
Жnon_trainable_variables
Я__call__
+р&call_and_return_all_conditional_losses
'р"call_and_return_conditional_losses"
_generic_user_object
>:<ђ2#separable_conv2d_7/depthwise_kernel
?:=ђђ2#separable_conv2d_7/pointwise_kernel
&:$ђ2separable_conv2d_7/bias
5
n0
o1
p2"
trackable_list_wrapper
5
n0
o1
p2"
trackable_list_wrapper
 "
trackable_list_wrapper
А
вlayers
q	variables
rtrainable_variables
sregularization_losses
 Вlayer_regularization_losses
ьmetrics
Ьnon_trainable_variables
Р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(ђ2batch_normalization_7/gamma
):'ђ2batch_normalization_7/beta
2:0ђ (2!batch_normalization_7/moving_mean
6:4ђ (2%batch_normalization_7/moving_variance
<
v0
w1
x2
y3"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
А
№layers
z	variables
{trainable_variables
|regularization_losses
 ­layer_regularization_losses
ыmetrics
Ыnon_trainable_variables
С__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
б
зlayers
~	variables
trainable_variables
ђregularization_losses
 Зlayer_regularization_losses
шmetrics
Шnon_trainable_variables
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
эlayers
ѓ	variables
Ѓtrainable_variables
ёregularization_losses
 Эlayer_regularization_losses
щmetrics
Щnon_trainable_variables
У__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
чlayers
є	variables
Єtrainable_variables
ѕregularization_losses
 Чlayer_regularization_losses
§metrics
■non_trainable_variables
Ж__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
": 
ђђ2dense_4/kernel
:ђ2dense_4/bias
0
і0
І1"
trackable_list_wrapper
0
і0
І1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
 layers
ї	variables
Їtrainable_variables
јregularization_losses
 ђlayer_regularization_losses
Ђmetrics
ѓnon_trainable_variables
В__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Ѓlayers
љ	variables
Љtrainable_variables
њregularization_losses
 ёlayer_regularization_losses
Ёmetrics
єnon_trainable_variables
Ь__call__
+№&call_and_return_all_conditional_losses
'№"call_and_return_conditional_losses"
_generic_user_object
": 
ђђ2dense_5/kernel
:ђ2dense_5/bias
0
ћ0
Ћ1"
trackable_list_wrapper
0
ћ0
Ћ1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Єlayers
ќ	variables
Ќtrainable_variables
ўregularization_losses
 ѕlayer_regularization_losses
Ѕmetrics
іnon_trainable_variables
­__call__
+ы&call_and_return_all_conditional_losses
'ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Іlayers
џ	variables
Џtrainable_variables
юregularization_losses
 їlayer_regularization_losses
Їmetrics
јnon_trainable_variables
Ы__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
!:	ђ@2dense_6/kernel
:@2dense_6/bias
0
ъ0
Ъ1"
trackable_list_wrapper
0
ъ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Јlayers
а	variables
Аtrainable_variables
бregularization_losses
 љlayer_regularization_losses
Љmetrics
њnon_trainable_variables
З__call__
+ш&call_and_return_all_conditional_losses
'ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Њlayers
ц	variables
Цtrainable_variables
дregularization_losses
 ћlayer_regularization_losses
Ћmetrics
ќnon_trainable_variables
Ш__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_7/kernel
:2dense_7/bias
0
е0
Е1"
trackable_list_wrapper
0
е0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Ќlayers
ф	variables
Фtrainable_variables
гregularization_losses
 ўlayer_regularization_losses
Ўmetrics
џnon_trainable_variables
Э__call__
+щ&call_and_return_all_conditional_losses
'щ"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
о
0
1
2
3
4
5
6
	7

8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23"
trackable_list_wrapper
 "
trackable_list_wrapper
(
Џ0"
trackable_list_wrapper
X
80
91
L2
M3
`4
a5
x6
y7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Б

юtotal

Юcount
ъ
_fn_kwargs
Ъ	variables
аtrainable_variables
Аregularization_losses
б	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"т
_tf_keras_layer╦{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ю0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
Бlayers
Ъ	variables
аtrainable_variables
Аregularization_losses
 цlayer_regularization_losses
Цmetrics
дnon_trainable_variables
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ю0
Ю1"
trackable_list_wrapper
3:12RMSprop/conv2d_1/kernel/rms
%:#2RMSprop/conv2d_1/bias/rms
G:E2/RMSprop/separable_conv2d_4/depthwise_kernel/rms
G:E 2/RMSprop/separable_conv2d_4/pointwise_kernel/rms
/:- 2#RMSprop/separable_conv2d_4/bias/rms
3:1 2'RMSprop/batch_normalization_4/gamma/rms
2:0 2&RMSprop/batch_normalization_4/beta/rms
G:E 2/RMSprop/separable_conv2d_5/depthwise_kernel/rms
G:E @2/RMSprop/separable_conv2d_5/pointwise_kernel/rms
/:-@2#RMSprop/separable_conv2d_5/bias/rms
3:1@2'RMSprop/batch_normalization_5/gamma/rms
2:0@2&RMSprop/batch_normalization_5/beta/rms
G:E@2/RMSprop/separable_conv2d_6/depthwise_kernel/rms
H:F@ђ2/RMSprop/separable_conv2d_6/pointwise_kernel/rms
0:.ђ2#RMSprop/separable_conv2d_6/bias/rms
4:2ђ2'RMSprop/batch_normalization_6/gamma/rms
3:1ђ2&RMSprop/batch_normalization_6/beta/rms
H:Fђ2/RMSprop/separable_conv2d_7/depthwise_kernel/rms
I:Gђђ2/RMSprop/separable_conv2d_7/pointwise_kernel/rms
0:.ђ2#RMSprop/separable_conv2d_7/bias/rms
4:2ђ2'RMSprop/batch_normalization_7/gamma/rms
3:1ђ2&RMSprop/batch_normalization_7/beta/rms
,:*
ђђ2RMSprop/dense_4/kernel/rms
%:#ђ2RMSprop/dense_4/bias/rms
,:*
ђђ2RMSprop/dense_5/kernel/rms
%:#ђ2RMSprop/dense_5/bias/rms
+:)	ђ@2RMSprop/dense_6/kernel/rms
$:"@2RMSprop/dense_6/bias/rms
*:(@2RMSprop/dense_7/kernel/rms
$:"2RMSprop/dense_7/bias/rms
є2Ѓ
.__inference_sequential_1_layer_call_fn_2657459
.__inference_sequential_1_layer_call_fn_2657347
.__inference_sequential_1_layer_call_fn_2658129
.__inference_sequential_1_layer_call_fn_2658086└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657236
I__inference_sequential_1_layer_call_and_return_conditional_losses_2658043
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657872
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657168└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
"__inference__wrapped_model_2655652┼
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *5б2
0і-
conv2d_1_input         dd
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Ѕ2є
*__inference_conv2d_1_layer_call_fn_2655677О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
ц2А
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
Ў2ќ
1__inference_max_pooling2d_5_layer_call_fn_2655694Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Њ2љ
4__inference_separable_conv2d_4_layer_call_fn_2655725О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
«2Ф
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           
ъ2Џ
7__inference_batch_normalization_4_layer_call_fn_2658286
7__inference_batch_normalization_4_layer_call_fn_2658219
7__inference_batch_normalization_4_layer_call_fn_2658210
7__inference_batch_normalization_4_layer_call_fn_2658295┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658255
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658277
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658179
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658201┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_6_layer_call_fn_2655884Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Њ2љ
4__inference_separable_conv2d_5_layer_call_fn_2655915О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
«2Ф
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                            
ъ2Џ
7__inference_batch_normalization_5_layer_call_fn_2658461
7__inference_batch_normalization_5_layer_call_fn_2658452
7__inference_batch_normalization_5_layer_call_fn_2658385
7__inference_batch_normalization_5_layer_call_fn_2658376┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658443
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658345
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658367
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658421┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_7_layer_call_fn_2656074Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Њ2љ
4__inference_separable_conv2d_6_layer_call_fn_2656105О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
«2Ф
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
ъ2Џ
7__inference_batch_normalization_6_layer_call_fn_2658542
7__inference_batch_normalization_6_layer_call_fn_2658551
7__inference_batch_normalization_6_layer_call_fn_2658627
7__inference_batch_normalization_6_layer_call_fn_2658618┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658587
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658511
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658533
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658609┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_8_layer_call_fn_2656264Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ћ2Љ
+__inference_dropout_5_layer_call_fn_2658657
+__inference_dropout_5_layer_call_fn_2658662┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658652
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658647┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ћ2Љ
4__inference_separable_conv2d_7_layer_call_fn_2656295п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
»2г
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283п
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *8б5
3і0,                           ђ
ъ2Џ
7__inference_batch_normalization_7_layer_call_fn_2658819
7__inference_batch_normalization_7_layer_call_fn_2658743
7__inference_batch_normalization_7_layer_call_fn_2658828
7__inference_batch_normalization_7_layer_call_fn_2658752┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
і2Є
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658810
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658788
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658734
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658712┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
1__inference_max_pooling2d_9_layer_call_fn_2656454Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
┤2▒
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
ћ2Љ
+__inference_dropout_6_layer_call_fn_2658858
+__inference_dropout_6_layer_call_fn_2658863┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658848
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658853┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Н2м
+__inference_flatten_1_layer_call_fn_2658874б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
­2ь
F__inference_flatten_1_layer_call_and_return_conditional_losses_2658869б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_4_layer_call_fn_2658892б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_4_layer_call_and_return_conditional_losses_2658885б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_7_layer_call_fn_2658927
+__inference_dropout_7_layer_call_fn_2658922┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658917
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658912┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_5_layer_call_fn_2658945б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_5_layer_call_and_return_conditional_losses_2658938б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_8_layer_call_fn_2658975
+__inference_dropout_8_layer_call_fn_2658980┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658965
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658970┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_6_layer_call_fn_2658998б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_6_layer_call_and_return_conditional_losses_2658991б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ћ2Љ
+__inference_dropout_9_layer_call_fn_2659028
+__inference_dropout_9_layer_call_fn_2659033┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╩2К
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659018
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659023┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
М2л
)__inference_dense_7_layer_call_fn_2659051б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_7_layer_call_and_return_conditional_losses_2659044б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
;B9
%__inference_signature_wrapper_2657568conv2d_1_input
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 г
F__inference_flatten_1_layer_call_and_return_conditional_losses_2658869b8б5
.б+
)і&
inputs         ђ
ф "&б#
і
0         ђ
џ Ь
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657236а.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕGбD
=б:
0і-
conv2d_1_input         dd
p 

 
ф "%б"
і
0         
џ К
1__inference_max_pooling2d_7_layer_call_fn_2656074ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╚
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658201r6789;б8
1б.
(і%
inputs         22 
p 
ф "-б*
#і 
0         22 
џ ђ
+__inference_dropout_7_layer_call_fn_2658922Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђб
7__inference_batch_normalization_7_layer_call_fn_2658743gvwxy<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђь
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658421ќJKLMMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ╚
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658367rJKLM;б8
1б.
(і%
inputs         @
p 
ф "-б*
#і 
0         @
џ й
4__inference_separable_conv2d_5_layer_call_fn_2655915ёBCDIбF
?б<
:і7
inputs+                            
ф "2і/+                           @№
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658587ў^_`aNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ б
7__inference_batch_normalization_7_layer_call_fn_2658752gvwxy<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђб
7__inference_batch_normalization_6_layer_call_fn_2658542g^_`a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђђ
+__inference_dropout_7_layer_call_fn_2658927Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђК
1__inference_max_pooling2d_6_layer_call_fn_2655884ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    б
7__inference_batch_normalization_6_layer_call_fn_2658551g^_`a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђ┐
4__inference_separable_conv2d_7_layer_call_fn_2656295єnopJбG
@б=
;і8
inputs,                           ђ
ф "3і0,                           ђь
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658443ќJKLMMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ т
O__inference_separable_conv2d_5_layer_call_and_return_conditional_losses_2655903ЉBCDIбF
?б<
:і7
inputs+                            
ф "?б<
5і2
0+                           @
џ й
4__inference_separable_conv2d_4_layer_call_fn_2655725ё./0IбF
?б<
:і7
inputs+                           
ф "2і/+                            у
O__inference_separable_conv2d_7_layer_call_and_return_conditional_losses_2656283ЊnopJбG
@б=
;і8
inputs,                           ђ
ф "@б=
6і3
0,                           ђ
џ Я
%__inference_signature_wrapper_2657568Х.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕQбN
б 
GфD
B
conv2d_1_input0і-
conv2d_1_input         dd"1ф.
,
dense_7!і
dense_7         К
1__inference_max_pooling2d_5_layer_call_fn_2655694ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    К
7__inference_batch_normalization_7_layer_call_fn_2658819ІvwxyNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђ╚
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658179r6789;б8
1б.
(і%
inputs         22 
p
ф "-б*
#і 
0         22 
џ Й
.__inference_sequential_1_layer_call_fn_2658129І.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕ?б<
5б2
(і%
inputs         dd
p 

 
ф "і         К
7__inference_batch_normalization_6_layer_call_fn_2658618І^_`aNбK
DбA
;і8
inputs,                           ђ
p
ф "3і0,                           ђК
7__inference_batch_normalization_7_layer_call_fn_2658828ІvwxyNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђЙ
.__inference_sequential_1_layer_call_fn_2658086І.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕ?б<
5б2
(і%
inputs         dd
p

 
ф "і         а
7__inference_batch_normalization_4_layer_call_fn_2658210e6789;б8
1б.
(і%
inputs         22 
p
ф " і         22 К
7__inference_batch_normalization_6_layer_call_fn_2658627І^_`aNбK
DбA
;і8
inputs,                           ђ
p 
ф "3і0,                           ђе
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658912^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ Т
O__inference_separable_conv2d_6_layer_call_and_return_conditional_losses_2656093њVWXIбF
?б<
:і7
inputs+                           @
ф "@б=
6і3
0,                           ђ
џ т
O__inference_separable_conv2d_4_layer_call_and_return_conditional_losses_2655713Љ./0IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                            
џ а
7__inference_batch_normalization_5_layer_call_fn_2658376eJKLM;б8
1б.
(і%
inputs         @
p
ф " і         @ь
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658255ќ6789MбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ▓
*__inference_conv2d_1_layer_call_fn_2655677Ѓ$%IбF
?б<
:і7
inputs+                           
ф "2і/+                           е
F__inference_dropout_7_layer_call_and_return_conditional_losses_2658917^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ а
7__inference_batch_normalization_4_layer_call_fn_2658219e6789;б8
1б.
(і%
inputs         22 
p 
ф " і         22 а
7__inference_batch_normalization_5_layer_call_fn_2658385eJKLM;б8
1б.
(і%
inputs         @
p 
ф " і         @~
)__inference_dense_7_layer_call_fn_2659051QеЕ/б,
%б"
 і
inputs         @
ф "і         д
D__inference_dense_7_layer_call_and_return_conditional_losses_2659044^еЕ/б,
%б"
 і
inputs         @
ф "%б"
і
0         
џ Т
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657872ў.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕ?б<
5б2
(і%
inputs         dd
p

 
ф "%б"
і
0         
џ ђ
)__inference_dense_4_layer_call_fn_2658892SіІ0б-
&б#
!і
inputs         ђ
ф "і         ђ┼
7__inference_batch_normalization_5_layer_call_fn_2658452ЅJKLMMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ь
R__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2658277ќ6789MбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ┼
7__inference_batch_normalization_5_layer_call_fn_2658461ЅJKLMMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @ё
+__inference_flatten_1_layer_call_fn_2658874U8б5
.б+
)і&
inputs         ђ
ф "і         ђљ
+__inference_dropout_6_layer_call_fn_2658858a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђљ
+__inference_dropout_6_layer_call_fn_2658863a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђе
D__inference_dense_4_layer_call_and_return_conditional_losses_2658885`іІ0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ ~
+__inference_dropout_9_layer_call_fn_2659028O3б0
)б&
 і
inputs         @
p
ф "і         @~
+__inference_dropout_9_layer_call_fn_2659033O3б0
)б&
 і
inputs         @
p 
ф "і         @љ
+__inference_dropout_5_layer_call_fn_2658657a<б9
2б/
)і&
inputs         ђ
p
ф "!і         ђљ
+__inference_dropout_5_layer_call_fn_2658662a<б9
2б/
)і&
inputs         ђ
p 
ф "!і         ђ┼
7__inference_batch_normalization_4_layer_call_fn_2658286Ѕ6789MбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            ┼
7__inference_batch_normalization_4_layer_call_fn_2658295Ѕ6789MбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            №
L__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_2656445ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ д
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659018\3б0
)б&
 і
inputs         @
p
ф "%б"
і
0         @
џ д
F__inference_dropout_9_layer_call_and_return_conditional_losses_2659023\3б0
)б&
 і
inputs         @
p 
ф "%б"
і
0         @
џ Т
I__inference_sequential_1_layer_call_and_return_conditional_losses_2658043ў.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕ?б<
5б2
(і%
inputs         dd
p 

 
ф "%б"
і
0         
џ И
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658848n<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ И
F__inference_dropout_6_layer_call_and_return_conditional_losses_2658853n<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ ┌
E__inference_conv2d_1_layer_call_and_return_conditional_losses_2655666љ$%IбF
?б<
:і7
inputs+                           
ф "?б<
5і2
0+                           
џ к
.__inference_sequential_1_layer_call_fn_2657347Њ.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕGбD
=б:
0і-
conv2d_1_input         dd
p

 
ф "і         ╩
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658712tvwxy<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ №
L__inference_max_pooling2d_8_layer_call_and_return_conditional_losses_2656255ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ 
)__inference_dense_6_layer_call_fn_2658998RъЪ0б-
&б#
!і
inputs         ђ
ф "і         @И
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658647n<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ И
F__inference_dropout_5_layer_call_and_return_conditional_losses_2658652n<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ Д
D__inference_dense_6_layer_call_and_return_conditional_losses_2658991_ъЪ0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         @
џ ╩
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658511t^_`a<б9
2б/
)і&
inputs         ђ
p
ф ".б+
$і!
0         ђ
џ ╩
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658734tvwxy<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ №
L__inference_max_pooling2d_7_layer_call_and_return_conditional_losses_2656065ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ╩
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658533t^_`a<б9
2б/
)і&
inputs         ђ
p 
ф ".б+
$і!
0         ђ
џ ђ
+__inference_dropout_8_layer_call_fn_2658975Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђђ
+__inference_dropout_8_layer_call_fn_2658980Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ№
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658810ўvwxyNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ №
L__inference_max_pooling2d_6_layer_call_and_return_conditional_losses_2655875ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ К
1__inference_max_pooling2d_9_layer_call_fn_2656454ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    к
.__inference_sequential_1_layer_call_fn_2657459Њ.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕGбD
=б:
0і-
conv2d_1_input         dd
p 

 
ф "і         ђ
)__inference_dense_5_layer_call_fn_2658945SћЋ0б-
&б#
!і
inputs         ђ
ф "і         ђК
1__inference_max_pooling2d_8_layer_call_fn_2656264ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ╚
R__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2658345rJKLM;б8
1б.
(і%
inputs         @
p
ф "-б*
#і 
0         @
џ №
R__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2658609ў^_`aNбK
DбA
;і8
inputs,                           ђ
p 
ф "@б=
6і3
0,                           ђ
џ Ь
I__inference_sequential_1_layer_call_and_return_conditional_losses_2657168а.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕGбD
=б:
0і-
conv2d_1_input         dd
p

 
ф "%б"
і
0         
џ №
L__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_2655685ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ е
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658970^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ е
F__inference_dropout_8_layer_call_and_return_conditional_losses_2658965^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ №
R__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2658788ўvwxyNбK
DбA
;і8
inputs,                           ђ
p
ф "@б=
6і3
0,                           ђ
џ ╦
"__inference__wrapped_model_2655652ц.$%./06789BCDJKLMVWX^_`anopvwxyіІћЋъЪеЕ?б<
5б2
0і-
conv2d_1_input         dd
ф "1ф.
,
dense_7!і
dense_7         Й
4__inference_separable_conv2d_6_layer_call_fn_2656105ЁVWXIбF
?б<
:і7
inputs+                           @
ф "3і0,                           ђе
D__inference_dense_5_layer_call_and_return_conditional_losses_2658938`ћЋ0б-
&б#
!і
inputs         ђ
ф "&б#
і
0         ђ
џ 