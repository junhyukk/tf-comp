֤0
??
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
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(?
.
Identity

input"T
output"T"	
Ttype
9
	IdentityN

input2T
output2T"
T
list(type)(0
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
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
3
Square
x"T
y"T"
Ttype:
2
	
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
executor_typestring ?
?
StatelessIf
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.0-dev202103112v1.12.1-52612-g74d34665fc18??+
q
layer_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_0/bias
j
 layer_0/bias/Read/ReadVariableOpReadVariableOplayer_0/bias*
_output_shapes	
:?*
dtype0
?
layer_0/igdn_0/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_0/igdn_0/reparam_beta
?
/layer_0/igdn_0/reparam_beta/Read/ReadVariableOpReadVariableOplayer_0/igdn_0/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_0/igdn_0/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namelayer_0/igdn_0/reparam_gamma
?
0layer_0/igdn_0/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_0/igdn_0/reparam_gamma* 
_output_shapes
:
??*
dtype0
?
layer_0/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namelayer_0/kernel_rdft
}
'layer_0/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_0/kernel_rdft* 
_output_shapes
:
??*
dtype0
q
layer_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_1/bias
j
 layer_1/bias/Read/ReadVariableOpReadVariableOplayer_1/bias*
_output_shapes	
:?*
dtype0
?
layer_1/igdn_1/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_1/igdn_1/reparam_beta
?
/layer_1/igdn_1/reparam_beta/Read/ReadVariableOpReadVariableOplayer_1/igdn_1/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_1/igdn_1/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namelayer_1/igdn_1/reparam_gamma
?
0layer_1/igdn_1/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_1/igdn_1/reparam_gamma* 
_output_shapes
:
??*
dtype0
?
layer_1/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namelayer_1/kernel_rdft
}
'layer_1/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_1/kernel_rdft* 
_output_shapes
:
??*
dtype0
q
layer_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_2/bias
j
 layer_2/bias/Read/ReadVariableOpReadVariableOplayer_2/bias*
_output_shapes	
:?*
dtype0
?
layer_2/igdn_2/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namelayer_2/igdn_2/reparam_beta
?
/layer_2/igdn_2/reparam_beta/Read/ReadVariableOpReadVariableOplayer_2/igdn_2/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_2/igdn_2/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*-
shared_namelayer_2/igdn_2/reparam_gamma
?
0layer_2/igdn_2/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_2/igdn_2/reparam_gamma* 
_output_shapes
:
??*
dtype0
?
layer_2/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namelayer_2/kernel_rdft
}
'layer_2/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_2/kernel_rdft* 
_output_shapes
:
??*
dtype0
p
layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namelayer_3/bias
i
 layer_3/bias/Read/ReadVariableOpReadVariableOplayer_3/bias*
_output_shapes
:*
dtype0
?
layer_3/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namelayer_3/kernel_rdft
|
'layer_3/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_3/kernel_rdft*
_output_shapes
:	?*
dtype0
?
ConstConst*
_output_shapes

:*
dtype0*?
value?B?"???L>?А>    ?А>    ?А>???>    ???>                        ?А>???>    ???>                        ??L>t ?=J????Pj??=*??А>?%?=??¾ʯ????p?                    ?А>?%?=??¾ʯ????p?                    ??L>?Pj??=*?t ?=J??>?А>ʯ????p??%?=???>                    ?А>ʯ????p??%?=???>                    ??L>?Pj??=*>t ?=J????А>ʯ????p>?%?=??¾                    ?А>ʯ????p>?%?=??¾                    ??L>t ?=J??>?Pj??=*>?А>?%?=???>ʯ????p>                    ?А>?%?=???>ʯ????p>                    ??L>?А>    ?А>    t ?=?%?=    ?%?=    J?????¾    ??¾    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>?Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>??L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>???Pj??>??B>??̽Г???=*???B>i?>?˔?.?d???L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>?Pj??>??B???̽Г?>?=*???B>i???˔?.?d>??L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d??Pj???̽Г???>??B??=*??˔?.?d???B>i????L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    t ?=?%?=    ?%?=    J??>???>    ???>    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>t ?=
t=??????̽?˔?J??>???=L>??Г??.?d???L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*???B>i?>?˔?.?d?t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>??L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*???B>i???˔?.?d>t ?=??̽?˔=
t=????J??>Г??.?d>???=L>????L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*??˔?.?d???B>i??t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>??L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    t ?=?%?=    ?%?=    J?????¾    ??¾    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i??t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>??L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>????L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d?t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>??L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d???L>?А>    ?А>    t ?=?%?=    ?%?=    J??>???>    ???>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J??>???=L>??Г??.?d??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i????L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>??L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J??>Г??.?d>???=L>???Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d???L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>
L
Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ?6
L
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *??:
L
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Const_7Const*
_output_shapes

:*
dtype0*?
value?B?"???L>?А>    ?А>    ?А>???>    ???>                        ?А>???>    ???>                        ??L>t ?=J????Pj??=*??А>?%?=??¾ʯ????p?                    ?А>?%?=??¾ʯ????p?                    ??L>?Pj??=*?t ?=J??>?А>ʯ????p??%?=???>                    ?А>ʯ????p??%?=???>                    ??L>?Pj??=*>t ?=J????А>ʯ????p>?%?=??¾                    ?А>ʯ????p>?%?=??¾                    ??L>t ?=J??>?Pj??=*>?А>?%?=???>ʯ????p>                    ?А>?%?=???>ʯ????p>                    ??L>?А>    ?А>    t ?=?%?=    ?%?=    J?????¾    ??¾    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>?Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>??L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>???Pj??>??B>??̽Г???=*???B>i?>?˔?.?d???L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>?Pj??>??B???̽Г?>?=*???B>i???˔?.?d>??L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d??Pj???̽Г???>??B??=*??˔?.?d???B>i????L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    t ?=?%?=    ?%?=    J??>???>    ???>    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>t ?=
t=??????̽?˔?J??>???=L>??Г??.?d???L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*???B>i?>?˔?.?d?t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>??L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*???B>i???˔?.?d>t ?=??̽?˔=
t=????J??>Г??.?d>???=L>????L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*??˔?.?d???B>i??t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>??L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    t ?=?%?=    ?%?=    J?????¾    ??¾    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i??t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>??L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>????L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d?t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>??L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d???L>?А>    ?А>    t ?=?%?=    ?%?=    J??>???>    ???>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J??>???=L>??Г??.?d??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i????L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>??L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J??>Г??.?d>???=L>???Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d???L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>
L
Const_8Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
L
Const_9Const*
_output_shapes
: *
dtype0*
valueB
 *  ?6
M
Const_10Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
M
Const_11Const*
_output_shapes
: *
dtype0*
valueB
 *??:
M
Const_12Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
M
Const_13Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Const_14Const*
_output_shapes

:*
dtype0*?
value?B?"???L>?А>    ?А>    ?А>???>    ???>                        ?А>???>    ???>                        ??L>t ?=J????Pj??=*??А>?%?=??¾ʯ????p?                    ?А>?%?=??¾ʯ????p?                    ??L>?Pj??=*?t ?=J??>?А>ʯ????p??%?=???>                    ?А>ʯ????p??%?=???>                    ??L>?Pj??=*>t ?=J????А>ʯ????p>?%?=??¾                    ?А>ʯ????p>?%?=??¾                    ??L>t ?=J??>?Pj??=*>?А>?%?=???>ʯ????p>                    ?А>?%?=???>ʯ????p>                    ??L>?А>    ?А>    t ?=?%?=    ?%?=    J?????¾    ??¾    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>?Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>??L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>???Pj??>??B>??̽Г???=*???B>i?>?˔?.?d???L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>?Pj??>??B???̽Г?>?=*???B>i???˔?.?d>??L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d??Pj???̽Г???>??B??=*??˔?.?d???B>i????L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    t ?=?%?=    ?%?=    J??>???>    ???>    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>t ?=
t=??????̽?˔?J??>???=L>??Г??.?d???L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*???B>i?>?˔?.?d?t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>??L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*???B>i???˔?.?d>t ?=??̽?˔=
t=????J??>Г??.?d>???=L>????L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*??˔?.?d???B>i??t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>??L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    t ?=?%?=    ?%?=    J?????¾    ??¾    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i??t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>??L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>????L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d?t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>??L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d???L>?А>    ?А>    t ?=?%?=    ?%?=    J??>???>    ???>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J??>???=L>??Г??.?d??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i????L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>??L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J??>Г??.?d>???=L>???Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d???L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>
M
Const_15Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
M
Const_16Const*
_output_shapes
: *
dtype0*
valueB
 *  ?6
M
Const_17Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
M
Const_18Const*
_output_shapes
: *
dtype0*
valueB
 *??:
M
Const_19Const*
_output_shapes
: *
dtype0*
valueB
 *  ?-
M
Const_20Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
Const_21Const*
_output_shapes

:*
dtype0*?
value?B?"???L>?А>    ?А>    ?А>???>    ???>                        ?А>???>    ???>                        ??L>t ?=J????Pj??=*??А>?%?=??¾ʯ????p?                    ?А>?%?=??¾ʯ????p?                    ??L>?Pj??=*?t ?=J??>?А>ʯ????p??%?=???>                    ?А>ʯ????p??%?=???>                    ??L>?Pj??=*>t ?=J????А>ʯ????p>?%?=??¾                    ?А>ʯ????p>?%?=??¾                    ??L>t ?=J??>?Pj??=*>?А>?%?=???>ʯ????p>                    ?А>?%?=???>ʯ????p>                    ??L>?А>    ?А>    t ?=?%?=    ?%?=    J?????¾    ??¾    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>?Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>??L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>???Pj??>??B>??̽Г???=*???B>i?>?˔?.?d???L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>?Pj??>??B???̽Г?>?=*???B>i???˔?.?d>??L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d??Pj???̽Г???>??B??=*??˔?.?d???B>i????L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*???p?    ??p?    t ?=?%?=    ?%?=    J??>???>    ???>    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*??˔?.?d>??B>i?>t ?=
t=??????̽?˔?J??>???=L>??Г??.?d???L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*???B>i?>?˔?.?d?t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>??L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*???B>i???˔?.?d>t ?=??̽?˔=
t=????J??>Г??.?d>???=L>????L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*??˔?.?d???B>i??t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>??L>?А>    ?А>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    t ?=?%?=    ?%?=    J?????¾    ??¾    ??L>t ?=J????Pj??=*??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i??t ?=
t=??????̽?˔?J???????L>?>Г?>.?d>??L>?Pj??=*?t ?=J??>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>t ?=??̽?˔?
t=???=J???Г?>.?d>????L>????L>?Pj??=*>t ?=J????Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d?t ?=??̽?˔=
t=????J???Г?>.?d?????L>?>??L>t ?=J??>?Pj??=*>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>t ?=
t=???=??̽?˔=J???????L>??Г?>.?d???L>?А>    ?А>    t ?=?%?=    ?%?=    J??>???>    ???>    ?Pj?ʯ??    ʯ??    ?=*>??p>    ??p>    ??L>t ?=J????Pj??=*?t ?=
t=??????̽?˔?J??>???=L>??Г??.?d??Pj???̽Г?>?>??B>?=*>?˔=.?d???B?i????L>?Pj??=*?t ?=J??>t ?=??̽?˔?
t=???=J??>Г??.?d????=L>?>?Pj??>??B>??̽Г???=*>??B?i???˔=.?d>??L>?Pj??=*>t ?=J???t ?=??̽?˔=
t=????J??>Г??.?d>???=L>???Pj??>??B???̽Г?>?=*>??B?i?>?˔=.?d???L>t ?=J??>?Pj??=*>t ?=
t=???=??̽?˔=J??>???=L>?>Г??.?d>?Pj???̽Г???>??B??=*>?˔=.?d>??B?i?>

NoOpNoOp
?,
Const_22Const"/device:CPU:0*
_output_shapes
: *
dtype0*?+
value?+B?+ B?+
?
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
?
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
layer-4
trainable_variables
regularization_losses
	variables
	keras_api
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
?

layers
 layer_regularization_losses
!non_trainable_variables
trainable_variables
"metrics
regularization_losses
	variables
#layer_metrics
 
?
$_activation
%_kernel_parameter
_bias_parameter
&trainable_variables
'regularization_losses
(	variables
)	keras_api
?
*_activation
+_kernel_parameter
_bias_parameter
,trainable_variables
-regularization_losses
.	variables
/	keras_api
?
0_activation
1_kernel_parameter
_bias_parameter
2trainable_variables
3regularization_losses
4	variables
5	keras_api
~
6_kernel_parameter
_bias_parameter
7trainable_variables
8regularization_losses
9	variables
:	keras_api
R
;trainable_variables
<regularization_losses
=	variables
>	keras_api
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
 
f
0
1
2
3
4
5
6
7
8
9
10
11
12
13
?

?layers
@layer_regularization_losses
Anon_trainable_variables
trainable_variables
Bmetrics
regularization_losses
	variables
Clayer_metrics
RP
VARIABLE_VALUElayer_0/bias0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElayer_0/igdn_0/reparam_beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_0/igdn_0/reparam_gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer_0/kernel_rdft0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer_1/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElayer_1/igdn_1/reparam_beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_1/igdn_1/reparam_gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer_1/kernel_rdft0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer_2/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElayer_2/igdn_2/reparam_beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUElayer_2/igdn_2/reparam_gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUElayer_2/kernel_rdft1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElayer_3/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUElayer_3/kernel_rdft1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 
 
 
}
D_beta_parameter
E_gamma_parameter
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api


rdft

0
1
2
3
 

0
1
2
3
?

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
&trainable_variables
'regularization_losses
(	variables
Nlayer_metrics
}
O_beta_parameter
P_gamma_parameter
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api


rdft

0
1
2
3
 

0
1
2
3
?

Ulayers
Vlayer_regularization_losses
Wnon_trainable_variables
Xmetrics
,trainable_variables
-regularization_losses
.	variables
Ylayer_metrics
}
Z_beta_parameter
[_gamma_parameter
\trainable_variables
]regularization_losses
^	variables
_	keras_api


rdft

0
1
2
3
 

0
1
2
3
?

`layers
alayer_regularization_losses
bnon_trainable_variables
cmetrics
2trainable_variables
3regularization_losses
4	variables
dlayer_metrics


rdft

0
1
 

0
1
?

elayers
flayer_regularization_losses
gnon_trainable_variables
hmetrics
7trainable_variables
8regularization_losses
9	variables
ilayer_metrics
 
 
 
?

jlayers
klayer_regularization_losses
lnon_trainable_variables
mmetrics
;trainable_variables
<regularization_losses
=	variables
nlayer_metrics
#
0
	1

2
3
4
 
 
 
 

variable

variable

0
1
 

0
1
?

olayers
player_regularization_losses
qnon_trainable_variables
rmetrics
Ftrainable_variables
Gregularization_losses
H	variables
slayer_metrics

$0
 
 
 
 

variable

variable

0
1
 

0
1
?

tlayers
ulayer_regularization_losses
vnon_trainable_variables
wmetrics
Qtrainable_variables
Rregularization_losses
S	variables
xlayer_metrics

*0
 
 
 
 

variable

variable

0
1
 

0
1
?

ylayers
zlayer_regularization_losses
{non_trainable_variables
|metrics
\trainable_variables
]regularization_losses
^	variables
}layer_metrics

00
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
?
serving_default_input_2Placeholder*B
_output_shapes0
.:,????????????????????????????*
dtype0*7
shape.:,????????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2Constlayer_0/kernel_rdftlayer_0/biasConst_1layer_0/igdn_0/reparam_gammaConst_2Const_3layer_0/igdn_0/reparam_betaConst_4Const_5Const_6Const_7layer_1/kernel_rdftlayer_1/biasConst_8layer_1/igdn_1/reparam_gammaConst_9Const_10layer_1/igdn_1/reparam_betaConst_11Const_12Const_13Const_14layer_2/kernel_rdftlayer_2/biasConst_15layer_2/igdn_2/reparam_gammaConst_16Const_17layer_2/igdn_2/reparam_betaConst_18Const_19Const_20Const_21layer_3/kernel_rdftlayer_3/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_202091
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename layer_0/bias/Read/ReadVariableOp/layer_0/igdn_0/reparam_beta/Read/ReadVariableOp0layer_0/igdn_0/reparam_gamma/Read/ReadVariableOp'layer_0/kernel_rdft/Read/ReadVariableOp layer_1/bias/Read/ReadVariableOp/layer_1/igdn_1/reparam_beta/Read/ReadVariableOp0layer_1/igdn_1/reparam_gamma/Read/ReadVariableOp'layer_1/kernel_rdft/Read/ReadVariableOp layer_2/bias/Read/ReadVariableOp/layer_2/igdn_2/reparam_beta/Read/ReadVariableOp0layer_2/igdn_2/reparam_gamma/Read/ReadVariableOp'layer_2/kernel_rdft/Read/ReadVariableOp layer_3/bias/Read/ReadVariableOp'layer_3/kernel_rdft/Read/ReadVariableOpConst_22*
Tin
2*
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
GPU2*0J 8? *(
f#R!
__inference__traced_save_204866
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_0/biaslayer_0/igdn_0/reparam_betalayer_0/igdn_0/reparam_gammalayer_0/kernel_rdftlayer_1/biaslayer_1/igdn_1/reparam_betalayer_1/igdn_1/reparam_gammalayer_1/kernel_rdftlayer_2/biaslayer_2/igdn_2/reparam_betalayer_2/igdn_2/reparam_gammalayer_2/kernel_rdftlayer_3/biaslayer_3/kernel_rdft*
Tin
2*
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
GPU2*0J 8? *+
f&R$
"__inference__traced_restore_204918??*
?
?
igdn_2_cond_2_cond_false_201044)
%igdn_2_cond_2_cond_pow_igdn_2_biasadd
igdn_2_cond_2_cond_pow_y
igdn_2_cond_2_cond_identity?
igdn_2/cond_2/cond/powPow%igdn_2_cond_2_cond_pow_igdn_2_biasaddigdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/pow?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity"C
igdn_2_cond_2_cond_identity$igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
h
layer_2_igdn_2_cond_true_203502#
layer_2_igdn_2_cond_placeholder
 
layer_2_igdn_2_cond_identity
x
layer_2/igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_2/igdn_2/cond/Const?
layer_2/igdn_2/cond/IdentityIdentity"layer_2/igdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity"E
layer_2_igdn_2_cond_identity%layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1model_synthesis_layer_0_igdn_0_cond_2_true_200128Y
Umodel_synthesis_layer_0_igdn_0_cond_2_identity_model_synthesis_layer_0_igdn_0_biasadd5
1model_synthesis_layer_0_igdn_0_cond_2_placeholder2
.model_synthesis_layer_0_igdn_0_cond_2_identity?
.model/synthesis/layer_0/igdn_0/cond_2/IdentityIdentityUmodel_synthesis_layer_0_igdn_0_cond_2_identity_model_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_2/Identity"i
.model_synthesis_layer_0_igdn_0_cond_2_identity7model/synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
<model_synthesis_layer_2_igdn_2_cond_1_cond_cond_false_200387W
Smodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_model_synthesis_layer_2_biasadd9
5model_synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y<
8model_synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
3model/synthesis/layer_2/igdn_2/cond_1/cond/cond/powPowSmodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_model_synthesis_layer_2_biasadd5model_synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_1/cond/cond/pow?
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity7model/synthesis/layer_2/igdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"}
8model_synthesis_layer_2_igdn_2_cond_1_cond_cond_identityAmodel/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
<model_synthesis_layer_0_igdn_0_cond_1_cond_cond_false_200065W
Smodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_model_synthesis_layer_0_biasadd9
5model_synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y<
8model_synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
3model/synthesis/layer_0/igdn_0/cond_1/cond/cond/powPowSmodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_model_synthesis_layer_0_biasadd5model_synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_1/cond/cond/pow?
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity7model/synthesis/layer_0/igdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"}
8model_synthesis_layer_0_igdn_0_cond_1_cond_cond_identityAmodel/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_0_igdn_0_cond_2_cond_true_203807:
6layer_0_igdn_0_cond_2_cond_sqrt_layer_0_igdn_0_biasadd*
&layer_0_igdn_0_cond_2_cond_placeholder'
#layer_0_igdn_0_cond_2_cond_identity?
layer_0/igdn_0/cond_2/cond/SqrtSqrt6layer_0_igdn_0_cond_2_cond_sqrt_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_0/igdn_0/cond_2/cond/Sqrt?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity#layer_0/igdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity"S
#layer_0_igdn_0_cond_2_cond_identity,layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_0_cond_false_204233%
!igdn_0_cond_identity_igdn_0_equal

igdn_0_cond_identity
|
igdn_0/cond/IdentityIdentity!igdn_0_cond_identity_igdn_0_equal*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity"5
igdn_0_cond_identityigdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
!layer_0_igdn_0_cond_2_true_2032749
5layer_0_igdn_0_cond_2_identity_layer_0_igdn_0_biasadd%
!layer_0_igdn_0_cond_2_placeholder"
layer_0_igdn_0_cond_2_identity?
layer_0/igdn_0/cond_2/IdentityIdentity5layer_0_igdn_0_cond_2_identity_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity"I
layer_0_igdn_0_cond_2_identity'layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_2_cond_1_true_200951"
igdn_2_cond_1_identity_biasadd
igdn_2_cond_1_placeholder
igdn_2_cond_1_identity?
igdn_2/cond_1/IdentityIdentityigdn_2_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity"9
igdn_2_cond_1_identityigdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_1_cond_false_204422#
igdn_1_cond_1_cond_cond_biasadd
igdn_1_cond_1_cond_equal_x
igdn_1_cond_1_cond_identityq
igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_1/cond_1/cond/x?
igdn_1/cond_1/cond/EqualEqualigdn_1_cond_1_cond_equal_xigdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_1/cond/Equal?
igdn_1/cond_1/cond/condStatelessIfigdn_1/cond_1/cond/Equal:z:0igdn_1_cond_1_cond_cond_biasaddigdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_1_cond_1_cond_cond_false_204432*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_1_cond_1_cond_cond_true_2044312
igdn_1/cond_1/cond/cond?
 igdn_1/cond_1/cond/cond/IdentityIdentity igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity?
igdn_1/cond_1/cond/IdentityIdentity)igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity"C
igdn_1_cond_1_cond_identity$igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
*__inference_synthesis_layer_call_fn_201548
layer_0_input
unknown
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3:
??
	unknown_4
	unknown_5
	unknown_6:	?
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:
??

unknown_12:	?

unknown_13

unknown_14:
??

unknown_15

unknown_16

unknown_17:	?

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22:
??

unknown_23:	?

unknown_24

unknown_25:
??

unknown_26

unknown_27

unknown_28:	?

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33:	?

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2014732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:q m
B
_output_shapes0
.:,????????????????????????????
'
_user_specified_namelayer_0_input:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
,layer_0_igdn_0_cond_1_cond_cond_false_2037357
3layer_0_igdn_0_cond_1_cond_cond_pow_layer_0_biasadd)
%layer_0_igdn_0_cond_1_cond_cond_pow_y,
(layer_0_igdn_0_cond_1_cond_cond_identity?
#layer_0/igdn_0/cond_1/cond/cond/powPow3layer_0_igdn_0_cond_1_cond_cond_pow_layer_0_biasadd%layer_0_igdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/cond/pow?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity'layer_0/igdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity"]
(layer_0_igdn_0_cond_1_cond_cond_identity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
<model_synthesis_layer_1_igdn_1_cond_1_cond_cond_false_200226W
Smodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_model_synthesis_layer_1_biasadd9
5model_synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y<
8model_synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
3model/synthesis/layer_1/igdn_1/cond_1/cond/cond/powPowSmodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_model_synthesis_layer_1_biasadd5model_synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_1/cond/cond/pow?
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity7model/synthesis/layer_1/igdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"}
8model_synthesis_layer_1_igdn_1_cond_1_cond_cond_identityAmodel/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_2_igdn_2_cond_1_cond_false_2040473
/layer_2_igdn_2_cond_1_cond_cond_layer_2_biasadd&
"layer_2_igdn_2_cond_1_cond_equal_x'
#layer_2_igdn_2_cond_1_cond_identity?
layer_2/igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_2/igdn_2/cond_1/cond/x?
 layer_2/igdn_2/cond_1/cond/EqualEqual"layer_2_igdn_2_cond_1_cond_equal_x%layer_2/igdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_2/igdn_2/cond_1/cond/Equal?
layer_2/igdn_2/cond_1/cond/condStatelessIf$layer_2/igdn_2/cond_1/cond/Equal:z:0/layer_2_igdn_2_cond_1_cond_cond_layer_2_biasadd"layer_2_igdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_2_igdn_2_cond_1_cond_cond_false_204057*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_2_igdn_2_cond_1_cond_cond_true_2040562!
layer_2/igdn_2/cond_1/cond/cond?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity(layer_2/igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity"S
#layer_2_igdn_2_cond_1_cond_identity,layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_1_igdn_1_cond_2_false_2034365
1layer_1_igdn_1_cond_2_cond_layer_1_igdn_1_biasadd!
layer_1_igdn_1_cond_2_equal_x"
layer_1_igdn_1_cond_2_identityw
layer_1/igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_1/igdn_1/cond_2/x?
layer_1/igdn_1/cond_2/EqualEquallayer_1_igdn_1_cond_2_equal_x layer_1/igdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/cond_2/Equal?
layer_1/igdn_1/cond_2/condStatelessIflayer_1/igdn_1/cond_2/Equal:z:01layer_1_igdn_1_cond_2_cond_layer_1_igdn_1_biasaddlayer_1_igdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_1_igdn_1_cond_2_cond_false_203445*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_2_cond_true_2034442
layer_1/igdn_1/cond_2/cond?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity#layer_1/igdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity?
layer_1/igdn_1/cond_2/IdentityIdentity,layer_1/igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity"I
layer_1_igdn_1_cond_2_identity'layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,layer_1_igdn_1_cond_1_cond_cond_false_2033727
3layer_1_igdn_1_cond_1_cond_cond_pow_layer_1_biasadd)
%layer_1_igdn_1_cond_1_cond_cond_pow_y,
(layer_1_igdn_1_cond_1_cond_cond_identity?
#layer_1/igdn_1/cond_1/cond/cond/powPow3layer_1_igdn_1_cond_1_cond_cond_pow_layer_1_biasadd%layer_1_igdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/cond/pow?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity'layer_1/igdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity"]
(layer_1_igdn_1_cond_1_cond_cond_identity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?i
?
C__inference_layer_0_layer_call_and_return_conditional_losses_204356

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y
igdn_0_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_0/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_0/x?
igdn_0/EqualEqualigdn_0_equal_xigdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/Equal?
igdn_0/condStatelessIfigdn_0/Equal:z:0igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_0_cond_false_204233*
output_shapes
: **
then_branchR
igdn_0_cond_true_2042322
igdn_0/condo
igdn_0/cond/IdentityIdentityigdn_0/cond:output:0*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity?
igdn_0/cond_1StatelessIfigdn_0/cond/Identity:output:0BiasAdd:output:0igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_0_cond_1_false_204244*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_1_true_2042432
igdn_0/cond_1?
igdn_0/cond_1/IdentityIdentityigdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204289*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204299*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_0/Reshape/shape?
igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_0/Reshape?
igdn_0/convolutionConv2Digdn_0/cond_1/Identity:output:0igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204313*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
igdn_0/BiasAddBiasAddigdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/BiasAdd]

igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_0/x_1?
igdn_0/Equal_1Equaligdn_0_equal_1_xigdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/Equal_1?
igdn_0/cond_2StatelessIfigdn_0/Equal_1:z:0igdn_0/BiasAdd:output:0igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_0_cond_2_false_204327*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_2_true_2043262
igdn_0/cond_2?
igdn_0/cond_2/IdentityIdentityigdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity?

igdn_0/mulMulBiasAdd:output:0igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_0/mul?
IdentityIdentityigdn_0/mul:z:0^BiasAdd/ReadVariableOp/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
2model_synthesis_layer_2_igdn_2_cond_2_false_200451U
Qmodel_synthesis_layer_2_igdn_2_cond_2_cond_model_synthesis_layer_2_igdn_2_biasadd1
-model_synthesis_layer_2_igdn_2_cond_2_equal_x2
.model_synthesis_layer_2_igdn_2_cond_2_identity?
'model/synthesis/layer_2/igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'model/synthesis/layer_2/igdn_2/cond_2/x?
+model/synthesis/layer_2/igdn_2/cond_2/EqualEqual-model_synthesis_layer_2_igdn_2_cond_2_equal_x0model/synthesis/layer_2/igdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_2/igdn_2/cond_2/Equal?
*model/synthesis/layer_2/igdn_2/cond_2/condStatelessIf/model/synthesis/layer_2/igdn_2/cond_2/Equal:z:0Qmodel_synthesis_layer_2_igdn_2_cond_2_cond_model_synthesis_layer_2_igdn_2_biasadd-model_synthesis_layer_2_igdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_2_igdn_2_cond_2_cond_false_200460*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_2_igdn_2_cond_2_cond_true_2004592,
*model/synthesis/layer_2/igdn_2/cond_2/cond?
3model/synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity3model/synthesis/layer_2/igdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_2/cond/Identity?
.model/synthesis/layer_2/igdn_2/cond_2/IdentityIdentity<model/synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_2/Identity"i
.model_synthesis_layer_2_igdn_2_cond_2_identity7model/synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_2_igdn_2_cond_1_false_203514.
*layer_2_igdn_2_cond_1_cond_layer_2_biasadd!
layer_2_igdn_2_cond_1_equal_x"
layer_2_igdn_2_cond_1_identityw
layer_2/igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/cond_1/x?
layer_2/igdn_2/cond_1/EqualEquallayer_2_igdn_2_cond_1_equal_x layer_2/igdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/cond_1/Equal?
layer_2/igdn_2/cond_1/condStatelessIflayer_2/igdn_2/cond_1/Equal:z:0*layer_2_igdn_2_cond_1_cond_layer_2_biasaddlayer_2_igdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_2_igdn_2_cond_1_cond_false_203523*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_1_cond_true_2035222
layer_2/igdn_2/cond_1/cond?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity#layer_2/igdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity?
layer_2/igdn_2/cond_1/IdentityIdentity,layer_2/igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity"I
layer_2_igdn_2_cond_1_identity'layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0model_synthesis_layer_2_igdn_2_cond_false_200357U
Qmodel_synthesis_layer_2_igdn_2_cond_identity_model_synthesis_layer_2_igdn_2_equal
0
,model_synthesis_layer_2_igdn_2_cond_identity
?
,model/synthesis/layer_2/igdn_2/cond/IdentityIdentityQmodel_synthesis_layer_2_igdn_2_cond_identity_model_synthesis_layer_2_igdn_2_equal*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_2/igdn_2/cond/Identity"e
,model_synthesis_layer_2_igdn_2_cond_identity5model/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
z
igdn_1_cond_2_false_200846%
!igdn_1_cond_2_cond_igdn_1_biasadd
igdn_1_cond_2_equal_x
igdn_1_cond_2_identityg
igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_1/cond_2/x?
igdn_1/cond_2/EqualEqualigdn_1_cond_2_equal_xigdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_2/Equal?
igdn_1/cond_2/condStatelessIfigdn_1/cond_2/Equal:z:0!igdn_1_cond_2_cond_igdn_1_biasaddigdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_1_cond_2_cond_false_200855*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_2_cond_true_2008542
igdn_1/cond_2/cond?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity?
igdn_1/cond_2/IdentityIdentity$igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity"9
igdn_1_cond_2_identityigdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
7model_synthesis_layer_0_igdn_0_cond_2_cond_false_200138Y
Umodel_synthesis_layer_0_igdn_0_cond_2_cond_pow_model_synthesis_layer_0_igdn_0_biasadd4
0model_synthesis_layer_0_igdn_0_cond_2_cond_pow_y7
3model_synthesis_layer_0_igdn_0_cond_2_cond_identity?
.model/synthesis/layer_0/igdn_0/cond_2/cond/powPowUmodel_synthesis_layer_0_igdn_0_cond_2_cond_pow_model_synthesis_layer_0_igdn_0_biasadd0model_synthesis_layer_0_igdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_2/cond/pow?
3model/synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity2model/synthesis/layer_0/igdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_2/cond/Identity"s
3model_synthesis_layer_0_igdn_0_cond_2_cond_identity<model/synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6model_synthesis_layer_2_igdn_2_cond_2_cond_true_200459Z
Vmodel_synthesis_layer_2_igdn_2_cond_2_cond_sqrt_model_synthesis_layer_2_igdn_2_biasadd:
6model_synthesis_layer_2_igdn_2_cond_2_cond_placeholder7
3model_synthesis_layer_2_igdn_2_cond_2_cond_identity?
/model/synthesis/layer_2/igdn_2/cond_2/cond/SqrtSqrtVmodel_synthesis_layer_2_igdn_2_cond_2_cond_sqrt_model_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/model/synthesis/layer_2/igdn_2/cond_2/cond/Sqrt?
3model/synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity3model/synthesis/layer_2/igdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_2/cond/Identity"s
3model_synthesis_layer_2_igdn_2_cond_2_cond_identity<model/synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_1_igdn_1_cond_1_cond_false_202314G
Csynthesis_layer_1_igdn_1_cond_1_cond_cond_synthesis_layer_1_biasadd0
,synthesis_layer_1_igdn_1_cond_1_cond_equal_x1
-synthesis_layer_1_igdn_1_cond_1_cond_identity?
&synthesis/layer_1/igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_1/igdn_1/cond_1/cond/x?
*synthesis/layer_1/igdn_1/cond_1/cond/EqualEqual,synthesis_layer_1_igdn_1_cond_1_cond_equal_x/synthesis/layer_1/igdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_1/igdn_1/cond_1/cond/Equal?
)synthesis/layer_1/igdn_1/cond_1/cond/condStatelessIf.synthesis/layer_1/igdn_1/cond_1/cond/Equal:z:0Csynthesis_layer_1_igdn_1_cond_1_cond_cond_synthesis_layer_1_biasadd,synthesis_layer_1_igdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202324*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_2023232+
)synthesis/layer_1/igdn_1/cond_1/cond/cond?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity2synthesis/layer_1/igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_1_cond_identity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_204749

inputs
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
mul/yu
mulMulinputsmul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
!layer_0_igdn_0_cond_1_true_2037152
.layer_0_igdn_0_cond_1_identity_layer_0_biasadd%
!layer_0_igdn_0_cond_1_placeholder"
layer_0_igdn_0_cond_1_identity?
layer_0/igdn_0/cond_1/IdentityIdentity.layer_0_igdn_0_cond_1_identity_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity"I
layer_0_igdn_0_cond_1_identity'layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_1_igdn_1_cond_1_true_2033522
.layer_1_igdn_1_cond_1_identity_layer_1_biasadd%
!layer_1_igdn_1_cond_1_placeholder"
layer_1_igdn_1_cond_1_identity?
layer_1/igdn_1/cond_1/IdentityIdentity.layer_1_igdn_1_cond_1_identity_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity"I
layer_1_igdn_1_cond_1_identity'layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+layer_2_igdn_2_cond_1_cond_cond_true_204056:
6layer_2_igdn_2_cond_1_cond_cond_square_layer_2_biasadd/
+layer_2_igdn_2_cond_1_cond_cond_placeholder,
(layer_2_igdn_2_cond_1_cond_cond_identity?
&layer_2/igdn_2/cond_1/cond/cond/SquareSquare6layer_2_igdn_2_cond_1_cond_cond_square_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_2/igdn_2/cond_1/cond/cond/Square?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity*layer_2/igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity"]
(layer_2_igdn_2_cond_1_cond_cond_identity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_2_cond_2_true_201034)
%igdn_2_cond_2_identity_igdn_2_biasadd
igdn_2_cond_2_placeholder
igdn_2_cond_2_identity?
igdn_2/cond_2/IdentityIdentity%igdn_2_cond_2_identity_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity"9
igdn_2_cond_2_identityigdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_202847N
Jsynthesis_layer_1_igdn_1_cond_1_cond_cond_square_synthesis_layer_1_biasadd9
5synthesis_layer_1_igdn_1_cond_1_cond_cond_placeholder6
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
0synthesis/layer_1/igdn_1/cond_1/cond/cond/SquareSquareJsynthesis_layer_1_igdn_1_cond_1_cond_cond_square_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_1/igdn_1/cond_1/cond/cond/Square?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity4synthesis/layer_1/igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"q
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_2_cond_false_200666)
%igdn_0_cond_2_cond_pow_igdn_0_biasadd
igdn_0_cond_2_cond_pow_y
igdn_0_cond_2_cond_identity?
igdn_0/cond_2/cond/powPow%igdn_0_cond_2_cond_pow_igdn_0_biasaddigdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/pow?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity"C
igdn_0_cond_2_cond_identity$igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
7model_synthesis_layer_1_igdn_1_cond_2_cond_false_200299Y
Umodel_synthesis_layer_1_igdn_1_cond_2_cond_pow_model_synthesis_layer_1_igdn_1_biasadd4
0model_synthesis_layer_1_igdn_1_cond_2_cond_pow_y7
3model_synthesis_layer_1_igdn_1_cond_2_cond_identity?
.model/synthesis/layer_1/igdn_1/cond_2/cond/powPowUmodel_synthesis_layer_1_igdn_1_cond_2_cond_pow_model_synthesis_layer_1_igdn_1_biasadd0model_synthesis_layer_1_igdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_2/cond/pow?
3model/synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity2model/synthesis/layer_1/igdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_2/cond/Identity"s
3model_synthesis_layer_1_igdn_1_cond_2_cond_identity<model/synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_2_true_204326)
%igdn_0_cond_2_identity_igdn_0_biasadd
igdn_0_cond_2_placeholder
igdn_0_cond_2_identity?
igdn_0/cond_2/IdentityIdentity%igdn_0_cond_2_identity_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity"9
igdn_0_cond_2_identityigdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_0_igdn_0_cond_2_cond_false_202760M
Isynthesis_layer_0_igdn_0_cond_2_cond_pow_synthesis_layer_0_igdn_0_biasadd.
*synthesis_layer_0_igdn_0_cond_2_cond_pow_y1
-synthesis_layer_0_igdn_0_cond_2_cond_identity?
(synthesis/layer_0/igdn_0/cond_2/cond/powPowIsynthesis_layer_0_igdn_0_cond_2_cond_pow_synthesis_layer_0_igdn_0_biasadd*synthesis_layer_0_igdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/cond/pow?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_2_cond_identity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+synthesis_layer_0_igdn_0_cond_2_true_202226M
Isynthesis_layer_0_igdn_0_cond_2_identity_synthesis_layer_0_igdn_0_biasadd/
+synthesis_layer_0_igdn_0_cond_2_placeholder,
(synthesis_layer_0_igdn_0_cond_2_identity?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentityIsynthesis_layer_0_igdn_0_cond_2_identity_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity"]
(synthesis_layer_0_igdn_0_cond_2_identity1synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_1_igdn_1_cond_1_cond_false_2033623
/layer_1_igdn_1_cond_1_cond_cond_layer_1_biasadd&
"layer_1_igdn_1_cond_1_cond_equal_x'
#layer_1_igdn_1_cond_1_cond_identity?
layer_1/igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_1/igdn_1/cond_1/cond/x?
 layer_1/igdn_1/cond_1/cond/EqualEqual"layer_1_igdn_1_cond_1_cond_equal_x%layer_1/igdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_1/igdn_1/cond_1/cond/Equal?
layer_1/igdn_1/cond_1/cond/condStatelessIf$layer_1/igdn_1/cond_1/cond/Equal:z:0/layer_1_igdn_1_cond_1_cond_cond_layer_1_biasadd"layer_1_igdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_1_igdn_1_cond_1_cond_cond_false_203372*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_1_igdn_1_cond_1_cond_cond_true_2033712!
layer_1/igdn_1/cond_1/cond/cond?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity(layer_1/igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity"S
#layer_1_igdn_1_cond_1_cond_identity,layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_0_cond_2_false_200657%
!igdn_0_cond_2_cond_igdn_0_biasadd
igdn_0_cond_2_equal_x
igdn_0_cond_2_identityg
igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_0/cond_2/x?
igdn_0/cond_2/EqualEqualigdn_0_cond_2_equal_xigdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_2/Equal?
igdn_0/cond_2/condStatelessIfigdn_0/cond_2/Equal:z:0!igdn_0_cond_2_cond_igdn_0_biasaddigdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_0_cond_2_cond_false_200666*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_2_cond_true_2006652
igdn_0/cond_2/cond?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity?
igdn_0/cond_2/IdentityIdentity$igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity"9
igdn_0_cond_2_identityigdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_0_igdn_0_cond_1_cond_true_202152F
Bsynthesis_layer_0_igdn_0_cond_1_cond_abs_synthesis_layer_0_biasadd4
0synthesis_layer_0_igdn_0_cond_1_cond_placeholder1
-synthesis_layer_0_igdn_0_cond_1_cond_identity?
(synthesis/layer_0/igdn_0/cond_1/cond/AbsAbsBsynthesis_layer_0_igdn_0_cond_1_cond_abs_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/cond/Abs?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_1_cond_identity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202324K
Gsynthesis_layer_1_igdn_1_cond_1_cond_cond_pow_synthesis_layer_1_biasadd3
/synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y6
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
-synthesis/layer_1/igdn_1/cond_1/cond/cond/powPowGsynthesis_layer_1_igdn_1_cond_1_cond_cond_pow_synthesis_layer_1_biasadd/synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/cond/pow?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity1synthesis/layer_1/igdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"q
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_2_true_200656)
%igdn_0_cond_2_identity_igdn_0_biasadd
igdn_0_cond_2_placeholder
igdn_0_cond_2_identity?
igdn_0/cond_2/IdentityIdentity%igdn_0_cond_2_identity_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity"9
igdn_0_cond_2_identityigdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_1_igdn_1_cond_1_cond_true_202837F
Bsynthesis_layer_1_igdn_1_cond_1_cond_abs_synthesis_layer_1_biasadd4
0synthesis_layer_1_igdn_1_cond_1_cond_placeholder1
-synthesis_layer_1_igdn_1_cond_1_cond_identity?
(synthesis/layer_1/igdn_1/cond_1/cond/AbsAbsBsynthesis_layer_1_igdn_1_cond_1_cond_abs_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/cond/Abs?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_1_cond_identity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_1_igdn_1_cond_2_cond_false_2034459
5layer_1_igdn_1_cond_2_cond_pow_layer_1_igdn_1_biasadd$
 layer_1_igdn_1_cond_2_cond_pow_y'
#layer_1_igdn_1_cond_2_cond_identity?
layer_1/igdn_1/cond_2/cond/powPow5layer_1_igdn_1_cond_2_cond_pow_layer_1_igdn_1_biasadd layer_1_igdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/cond/pow?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity"layer_1/igdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity"S
#layer_1_igdn_1_cond_2_cond_identity,layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_1_igdn_1_cond_2_cond_false_2039699
5layer_1_igdn_1_cond_2_cond_pow_layer_1_igdn_1_biasadd$
 layer_1_igdn_1_cond_2_cond_pow_y'
#layer_1_igdn_1_cond_2_cond_identity?
layer_1/igdn_1/cond_2/cond/powPow5layer_1_igdn_1_cond_2_cond_pow_layer_1_igdn_1_biasadd layer_1_igdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/cond/pow?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity"layer_1/igdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity"S
#layer_1_igdn_1_cond_2_cond_identity,layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_1_igdn_1_cond_2_cond_false_202397M
Isynthesis_layer_1_igdn_1_cond_2_cond_pow_synthesis_layer_1_igdn_1_biasadd.
*synthesis_layer_1_igdn_1_cond_2_cond_pow_y1
-synthesis_layer_1_igdn_1_cond_2_cond_identity?
(synthesis/layer_1/igdn_1/cond_2/cond/powPowIsynthesis_layer_1_igdn_1_cond_2_cond_pow_synthesis_layer_1_igdn_1_biasadd*synthesis_layer_1_igdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/cond/pow?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_2_cond_identity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_0_igdn_0_cond_2_cond_false_2032849
5layer_0_igdn_0_cond_2_cond_pow_layer_0_igdn_0_biasadd$
 layer_0_igdn_0_cond_2_cond_pow_y'
#layer_0_igdn_0_cond_2_cond_identity?
layer_0/igdn_0/cond_2/cond/powPow5layer_0_igdn_0_cond_2_cond_pow_layer_0_igdn_0_biasadd layer_0_igdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/cond/pow?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity"layer_0/igdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity"S
#layer_0_igdn_0_cond_2_cond_identity,layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_1_igdn_1_cond_2_true_2039599
5layer_1_igdn_1_cond_2_identity_layer_1_igdn_1_biasadd%
!layer_1_igdn_1_cond_2_placeholder"
layer_1_igdn_1_cond_2_identity?
layer_1/igdn_1/cond_2/IdentityIdentity5layer_1_igdn_1_cond_2_identity_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity"I
layer_1_igdn_1_cond_2_identity'layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_0_cond_1_true_200573"
igdn_0_cond_1_identity_biasadd
igdn_0_cond_1_placeholder
igdn_0_cond_1_identity?
igdn_0/cond_1/IdentityIdentityigdn_0_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity"9
igdn_0_cond_1_identityigdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_1_cond_true_200771"
igdn_1_cond_1_cond_abs_biasadd"
igdn_1_cond_1_cond_placeholder
igdn_1_cond_1_cond_identity?
igdn_1/cond_1/cond/AbsAbsigdn_1_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Abs?
igdn_1/cond_1/cond/IdentityIdentityigdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity"C
igdn_1_cond_1_cond_identity$igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
P
igdn_2_cond_true_200940
igdn_2_cond_placeholder

igdn_2_cond_identity
h
igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_2/cond/Constu
igdn_2/cond/IdentityIdentityigdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity"5
igdn_2_cond_identityigdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1synthesis_layer_2_igdn_2_cond_1_cond_false_202475G
Csynthesis_layer_2_igdn_2_cond_1_cond_cond_synthesis_layer_2_biasadd0
,synthesis_layer_2_igdn_2_cond_1_cond_equal_x1
-synthesis_layer_2_igdn_2_cond_1_cond_identity?
&synthesis/layer_2/igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_2/igdn_2/cond_1/cond/x?
*synthesis/layer_2/igdn_2/cond_1/cond/EqualEqual,synthesis_layer_2_igdn_2_cond_1_cond_equal_x/synthesis/layer_2/igdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_2/igdn_2/cond_1/cond/Equal?
)synthesis/layer_2/igdn_2/cond_1/cond/condStatelessIf.synthesis/layer_2/igdn_2/cond_1/cond/Equal:z:0Csynthesis_layer_2_igdn_2_cond_1_cond_cond_synthesis_layer_2_biasadd,synthesis_layer_2_igdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202485*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_2024842+
)synthesis/layer_2/igdn_2/cond_1/cond/cond?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity2synthesis/layer_2/igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_1_cond_identity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201473

inputs
layer_0_201395"
layer_0_201397:
??
layer_0_201399:	?
layer_0_201401"
layer_0_201403:
??
layer_0_201405
layer_0_201407
layer_0_201409:	?
layer_0_201411
layer_0_201413
layer_0_201415
layer_1_201418"
layer_1_201420:
??
layer_1_201422:	?
layer_1_201424"
layer_1_201426:
??
layer_1_201428
layer_1_201430
layer_1_201432:	?
layer_1_201434
layer_1_201436
layer_1_201438
layer_2_201441"
layer_2_201443:
??
layer_2_201445:	?
layer_2_201447"
layer_2_201449:
??
layer_2_201451
layer_2_201453
layer_2_201455:	?
layer_2_201457
layer_2_201459
layer_2_201461
layer_3_201464!
layer_3_201466:	?
layer_3_201468:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_201395layer_0_201397layer_0_201399layer_0_201401layer_0_201403layer_0_201405layer_0_201407layer_0_201409layer_0_201411layer_0_201413layer_0_201415*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_2006862!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201418layer_1_201420layer_1_201422layer_1_201424layer_1_201426layer_1_201428layer_1_201430layer_1_201432layer_1_201434layer_1_201436layer_1_201438*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_1_layer_call_and_return_conditional_losses_2008752!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201441layer_2_201443layer_2_201445layer_2_201447layer_2_201449layer_2_201451layer_2_201453layer_2_201455layer_2_201457layer_2_201459layer_2_201461*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_2_layer_call_and_return_conditional_losses_2010642!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201464layer_3_201466layer_3_201468*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_2011272!
layer_3/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2012282
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?	
?
;model_synthesis_layer_1_igdn_1_cond_1_cond_cond_true_200225Z
Vmodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_square_model_synthesis_layer_1_biasadd?
;model_synthesis_layer_1_igdn_1_cond_1_cond_cond_placeholder<
8model_synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
6model/synthesis/layer_1/igdn_1/cond_1/cond/cond/SquareSquareVmodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_square_model_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6model/synthesis/layer_1/igdn_1/cond_1/cond/cond/Square?
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity:model/synthesis/layer_1/igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"}
8model_synthesis_layer_1_igdn_1_cond_1_cond_cond_identityAmodel/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_2_igdn_2_cond_2_cond_false_202558M
Isynthesis_layer_2_igdn_2_cond_2_cond_pow_synthesis_layer_2_igdn_2_biasadd.
*synthesis_layer_2_igdn_2_cond_2_cond_pow_y1
-synthesis_layer_2_igdn_2_cond_2_cond_identity?
(synthesis/layer_2/igdn_2/cond_2/cond/powPowIsynthesis_layer_2_igdn_2_cond_2_cond_pow_synthesis_layer_2_igdn_2_biasadd*synthesis_layer_2_igdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/cond/pow?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_2_cond_identity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_202323N
Jsynthesis_layer_1_igdn_1_cond_1_cond_cond_square_synthesis_layer_1_biasadd9
5synthesis_layer_1_igdn_1_cond_1_cond_cond_placeholder6
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
0synthesis/layer_1/igdn_1/cond_1/cond/cond/SquareSquareJsynthesis_layer_1_igdn_1_cond_1_cond_cond_square_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_1/igdn_1/cond_1/cond/cond/Square?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity4synthesis/layer_1/igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"q
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
*synthesis_layer_1_igdn_1_cond_false_202818I
Esynthesis_layer_1_igdn_1_cond_identity_synthesis_layer_1_igdn_1_equal
*
&synthesis_layer_1_igdn_1_cond_identity
?
&synthesis/layer_1/igdn_1/cond/IdentityIdentityEsynthesis_layer_1_igdn_1_cond_identity_synthesis_layer_1_igdn_1_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity"Y
&synthesis_layer_1_igdn_1_cond_identity/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201231
layer_0_input
layer_0_201147"
layer_0_201149:
??
layer_0_201151:	?
layer_0_201153"
layer_0_201155:
??
layer_0_201157
layer_0_201159
layer_0_201161:	?
layer_0_201163
layer_0_201165
layer_0_201167
layer_1_201170"
layer_1_201172:
??
layer_1_201174:	?
layer_1_201176"
layer_1_201178:
??
layer_1_201180
layer_1_201182
layer_1_201184:	?
layer_1_201186
layer_1_201188
layer_1_201190
layer_2_201193"
layer_2_201195:
??
layer_2_201197:	?
layer_2_201199"
layer_2_201201:
??
layer_2_201203
layer_2_201205
layer_2_201207:	?
layer_2_201209
layer_2_201211
layer_2_201213
layer_3_201216!
layer_3_201218:	?
layer_3_201220:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputlayer_0_201147layer_0_201149layer_0_201151layer_0_201153layer_0_201155layer_0_201157layer_0_201159layer_0_201161layer_0_201163layer_0_201165layer_0_201167*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_2006862!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201170layer_1_201172layer_1_201174layer_1_201176layer_1_201178layer_1_201180layer_1_201182layer_1_201184layer_1_201186layer_1_201188layer_1_201190*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_1_layer_call_and_return_conditional_losses_2008752!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201193layer_2_201195layer_2_201197layer_2_201199layer_2_201201layer_2_201203layer_2_201205layer_2_201207layer_2_201209layer_2_201211layer_2_201213*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_2_layer_call_and_return_conditional_losses_2010642!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201216layer_3_201218layer_3_201220*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_2011272!
layer_3/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2012282
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:q m
B
_output_shapes0
.:,????????????????????????????
'
_user_specified_namelayer_0_input:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
$igdn_0_cond_1_cond_cond_false_200593'
#igdn_0_cond_1_cond_cond_pow_biasadd!
igdn_0_cond_1_cond_cond_pow_y$
 igdn_0_cond_1_cond_cond_identity?
igdn_0/cond_1/cond/cond/powPow#igdn_0_cond_1_cond_cond_pow_biasaddigdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/cond/pow?
 igdn_0/cond_1/cond/cond/IdentityIdentityigdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity"M
 igdn_0_cond_1_cond_cond_identity)igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_1_igdn_1_cond_2_cond_true_202396N
Jsynthesis_layer_1_igdn_1_cond_2_cond_sqrt_synthesis_layer_1_igdn_1_biasadd4
0synthesis_layer_1_igdn_1_cond_2_cond_placeholder1
-synthesis_layer_1_igdn_1_cond_2_cond_identity?
)synthesis/layer_1/igdn_1/cond_2/cond/SqrtSqrtJsynthesis_layer_1_igdn_1_cond_2_cond_sqrt_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_1/igdn_1/cond_2/cond/Sqrt?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_2_cond_identity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_1_igdn_1_cond_1_true_2038762
.layer_1_igdn_1_cond_1_identity_layer_1_biasadd%
!layer_1_igdn_1_cond_1_placeholder"
layer_1_igdn_1_cond_1_identity?
layer_1/igdn_1/cond_1/IdentityIdentity.layer_1_igdn_1_cond_1_identity_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity"I
layer_1_igdn_1_cond_1_identity'layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1model_synthesis_layer_2_igdn_2_cond_2_true_200450Y
Umodel_synthesis_layer_2_igdn_2_cond_2_identity_model_synthesis_layer_2_igdn_2_biasadd5
1model_synthesis_layer_2_igdn_2_cond_2_placeholder2
.model_synthesis_layer_2_igdn_2_cond_2_identity?
.model/synthesis/layer_2/igdn_2/cond_2/IdentityIdentityUmodel_synthesis_layer_2_igdn_2_cond_2_identity_model_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_2/Identity"i
.model_synthesis_layer_2_igdn_2_cond_2_identity7model/synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_2_igdn_2_cond_1_false_202466B
>synthesis_layer_2_igdn_2_cond_1_cond_synthesis_layer_2_biasadd+
'synthesis_layer_2_igdn_2_cond_1_equal_x,
(synthesis_layer_2_igdn_2_cond_1_identity?
!synthesis/layer_2/igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_2/igdn_2/cond_1/x?
%synthesis/layer_2/igdn_2/cond_1/EqualEqual'synthesis_layer_2_igdn_2_cond_1_equal_x*synthesis/layer_2/igdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_2/igdn_2/cond_1/Equal?
$synthesis/layer_2/igdn_2/cond_1/condStatelessIf)synthesis/layer_2/igdn_2/cond_1/Equal:z:0>synthesis_layer_2_igdn_2_cond_1_cond_synthesis_layer_2_biasadd'synthesis_layer_2_igdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_2_igdn_2_cond_1_cond_false_202475*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_1_cond_true_2024742&
$synthesis/layer_2/igdn_2/cond_1/cond?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity"]
(synthesis_layer_2_igdn_2_cond_1_identity1synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_1_igdn_1_cond_1_cond_true_2038852
.layer_1_igdn_1_cond_1_cond_abs_layer_1_biasadd*
&layer_1_igdn_1_cond_1_cond_placeholder'
#layer_1_igdn_1_cond_1_cond_identity?
layer_1/igdn_1/cond_1/cond/AbsAbs.layer_1_igdn_1_cond_1_cond_abs_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/cond/Abs?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity"layer_1/igdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity"S
#layer_1_igdn_1_cond_1_cond_identity,layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
7model_synthesis_layer_0_igdn_0_cond_1_cond_false_200055S
Omodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_model_synthesis_layer_0_biasadd6
2model_synthesis_layer_0_igdn_0_cond_1_cond_equal_x7
3model_synthesis_layer_0_igdn_0_cond_1_cond_identity?
,model/synthesis/layer_0/igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,model/synthesis/layer_0/igdn_0/cond_1/cond/x?
0model/synthesis/layer_0/igdn_0/cond_1/cond/EqualEqual2model_synthesis_layer_0_igdn_0_cond_1_cond_equal_x5model/synthesis/layer_0/igdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0model/synthesis/layer_0/igdn_0/cond_1/cond/Equal?
/model/synthesis/layer_0/igdn_0/cond_1/cond/condStatelessIf4model/synthesis/layer_0/igdn_0/cond_1/cond/Equal:z:0Omodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_model_synthesis_layer_0_biasadd2model_synthesis_layer_0_igdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *O
else_branch@R>
<model_synthesis_layer_0_igdn_0_cond_1_cond_cond_false_200065*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_0_igdn_0_cond_1_cond_cond_true_20006421
/model/synthesis/layer_0/igdn_0/cond_1/cond/cond?
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity8model/synthesis/layer_0/igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity?
3model/synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentityAmodel/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_1/cond/Identity"s
3model_synthesis_layer_0_igdn_0_cond_1_cond_identity<model/synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_2_cond_false_200941%
!igdn_2_cond_identity_igdn_2_equal

igdn_2_cond_identity
|
igdn_2/cond/IdentityIdentity!igdn_2_cond_identity_igdn_2_equal*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity"5
igdn_2_cond_identityigdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
0synthesis_layer_2_igdn_2_cond_1_cond_true_202998F
Bsynthesis_layer_2_igdn_2_cond_1_cond_abs_synthesis_layer_2_biasadd4
0synthesis_layer_2_igdn_2_cond_1_cond_placeholder1
-synthesis_layer_2_igdn_2_cond_1_cond_identity?
(synthesis/layer_2/igdn_2/cond_1/cond/AbsAbsBsynthesis_layer_2_igdn_2_cond_1_cond_abs_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/cond/Abs?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_1_cond_identity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
h
layer_0_igdn_0_cond_true_203704#
layer_0_igdn_0_cond_placeholder
 
layer_0_igdn_0_cond_identity
x
layer_0/igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_0/igdn_0/cond/Const?
layer_0/igdn_0/cond/IdentityIdentity"layer_0/igdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity"E
layer_0_igdn_0_cond_identity%layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
,synthesis_layer_1_igdn_1_cond_1_false_202305B
>synthesis_layer_1_igdn_1_cond_1_cond_synthesis_layer_1_biasadd+
'synthesis_layer_1_igdn_1_cond_1_equal_x,
(synthesis_layer_1_igdn_1_cond_1_identity?
!synthesis/layer_1/igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_1/igdn_1/cond_1/x?
%synthesis/layer_1/igdn_1/cond_1/EqualEqual'synthesis_layer_1_igdn_1_cond_1_equal_x*synthesis/layer_1/igdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_1/igdn_1/cond_1/Equal?
$synthesis/layer_1/igdn_1/cond_1/condStatelessIf)synthesis/layer_1/igdn_1/cond_1/Equal:z:0>synthesis_layer_1_igdn_1_cond_1_cond_synthesis_layer_1_biasadd'synthesis_layer_1_igdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_1_igdn_1_cond_1_cond_false_202314*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_1_cond_true_2023132&
$synthesis/layer_1/igdn_1/cond_1/cond?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity"]
(synthesis_layer_1_igdn_1_cond_1_identity1synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
)synthesis_layer_2_igdn_2_cond_true_202454-
)synthesis_layer_2_igdn_2_cond_placeholder
*
&synthesis_layer_2_igdn_2_cond_identity
?
#synthesis/layer_2/igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_2/igdn_2/cond/Const?
&synthesis/layer_2/igdn_2/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity"Y
&synthesis_layer_2_igdn_2_cond_identity/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?i
?
C__inference_layer_2_layer_call_and_return_conditional_losses_204694

inputs
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y
igdn_2_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_2/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_2/x?
igdn_2/EqualEqualigdn_2_equal_xigdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/Equal?
igdn_2/condStatelessIfigdn_2/Equal:z:0igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_2_cond_false_204571*
output_shapes
: **
then_branchR
igdn_2_cond_true_2045702
igdn_2/condo
igdn_2/cond/IdentityIdentityigdn_2/cond:output:0*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity?
igdn_2/cond_1StatelessIfigdn_2/cond/Identity:output:0BiasAdd:output:0igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_2_cond_1_false_204582*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_1_true_2045812
igdn_2/cond_1?
igdn_2/cond_1/IdentityIdentityigdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204627*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204637*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_2/Reshape/shape?
igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_2/Reshape?
igdn_2/convolutionConv2Digdn_2/cond_1/Identity:output:0igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204651*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
igdn_2/BiasAddBiasAddigdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/BiasAdd]

igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_2/x_1?
igdn_2/Equal_1Equaligdn_2_equal_1_xigdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/Equal_1?
igdn_2/cond_2StatelessIfigdn_2/Equal_1:z:0igdn_2/BiasAdd:output:0igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_2_cond_2_false_204665*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_2_true_2046642
igdn_2/cond_2?
igdn_2/cond_2/IdentityIdentityigdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity?

igdn_2/mulMulBiasAdd:output:0igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_2/mul?
IdentityIdentityigdn_2/mul:z:0^BiasAdd/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
igdn_2_cond_2_true_204664)
%igdn_2_cond_2_identity_igdn_2_biasadd
igdn_2_cond_2_placeholder
igdn_2_cond_2_identity?
igdn_2/cond_2/IdentityIdentity%igdn_2_cond_2_identity_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity"9
igdn_2_cond_2_identityigdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1model_synthesis_layer_0_igdn_0_cond_1_true_200045R
Nmodel_synthesis_layer_0_igdn_0_cond_1_identity_model_synthesis_layer_0_biasadd5
1model_synthesis_layer_0_igdn_0_cond_1_placeholder2
.model_synthesis_layer_0_igdn_0_cond_1_identity?
.model/synthesis/layer_0/igdn_0/cond_1/IdentityIdentityNmodel_synthesis_layer_0_igdn_0_cond_1_identity_model_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_1/Identity"i
.model_synthesis_layer_0_igdn_0_cond_1_identity7model/synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
)synthesis_layer_1_igdn_1_cond_true_202817-
)synthesis_layer_1_igdn_1_cond_placeholder
*
&synthesis_layer_1_igdn_1_cond_identity
?
#synthesis/layer_1/igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_1/igdn_1/cond/Const?
&synthesis/layer_1/igdn_1/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity"Y
&synthesis_layer_1_igdn_1_cond_identity/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
igdn_1_cond_2_cond_false_200855)
%igdn_1_cond_2_cond_pow_igdn_1_biasadd
igdn_1_cond_2_cond_pow_y
igdn_1_cond_2_cond_identity?
igdn_1/cond_2/cond/powPow%igdn_1_cond_2_cond_pow_igdn_1_biasaddigdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/pow?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity"C
igdn_1_cond_2_cond_identity$igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+layer_0_igdn_0_cond_1_cond_cond_true_203734:
6layer_0_igdn_0_cond_1_cond_cond_square_layer_0_biasadd/
+layer_0_igdn_0_cond_1_cond_cond_placeholder,
(layer_0_igdn_0_cond_1_cond_cond_identity?
&layer_0/igdn_0/cond_1/cond/cond/SquareSquare6layer_0_igdn_0_cond_1_cond_cond_square_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_0/igdn_0/cond_1/cond/cond/Square?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity*layer_0/igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity"]
(layer_0_igdn_0_cond_1_cond_cond_identity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_2_igdn_2_cond_2_false_2035975
1layer_2_igdn_2_cond_2_cond_layer_2_igdn_2_biasadd!
layer_2_igdn_2_cond_2_equal_x"
layer_2_igdn_2_cond_2_identityw
layer_2/igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_2/igdn_2/cond_2/x?
layer_2/igdn_2/cond_2/EqualEquallayer_2_igdn_2_cond_2_equal_x layer_2/igdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/cond_2/Equal?
layer_2/igdn_2/cond_2/condStatelessIflayer_2/igdn_2/cond_2/Equal:z:01layer_2_igdn_2_cond_2_cond_layer_2_igdn_2_biasaddlayer_2_igdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_2_igdn_2_cond_2_cond_false_203606*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_2_cond_true_2036052
layer_2/igdn_2/cond_2/cond?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity#layer_2/igdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity?
layer_2/igdn_2/cond_2/IdentityIdentity,layer_2/igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity"I
layer_2_igdn_2_cond_2_identity'layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_1_igdn_1_cond_false_2033425
1layer_1_igdn_1_cond_identity_layer_1_igdn_1_equal
 
layer_1_igdn_1_cond_identity
?
layer_1/igdn_1/cond/IdentityIdentity1layer_1_igdn_1_cond_identity_layer_1_igdn_1_equal*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity"E
layer_1_igdn_1_cond_identity%layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
??
?
E__inference_synthesis_layer_call_and_return_conditional_losses_204187

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??6
'layer_0_biasadd_readvariableop_resource:	?
layer_0_igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y
layer_0_igdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??6
'layer_1_biasadd_readvariableop_resource:	?
layer_1_igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y
layer_1_igdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??6
'layer_2_biasadd_readvariableop_resource:	?
layer_2_igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y
layer_2_igdn_2_equal_1_x
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	?5
'layer_3_biasadd_readvariableop_resource:
identity??layer_0/BiasAdd/ReadVariableOp?.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?layer_1/BiasAdd/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshape?
layer_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_0/transpose/perm?
layer_0/transpose	Transposelayer_0/kernel/Reshape:output:0layer_0/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_0/transposeT
layer_0/ShapeShapeinputs*
T0*
_output_shapes
:2
layer_0/Shape?
layer_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_0/strided_slice/stack?
layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice/stack_1?
layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice/stack_2?
layer_0/strided_sliceStridedSlicelayer_0/Shape:output:0$layer_0/strided_slice/stack:output:0&layer_0/strided_slice/stack_1:output:0&layer_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice?
layer_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice_1/stack?
layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_1/stack_1?
layer_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_1/stack_2?
layer_0/strided_slice_1StridedSlicelayer_0/Shape:output:0&layer_0/strided_slice_1/stack:output:0(layer_0/strided_slice_1/stack_1:output:0(layer_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice_1`
layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_0/mul/y|
layer_0/mulMul layer_0/strided_slice_1:output:0layer_0/mul/y:output:0*
T0*
_output_shapes
: 2
layer_0/mul`
layer_0/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_0/add/ym
layer_0/addAddV2layer_0/mul:z:0layer_0/add/y:output:0*
T0*
_output_shapes
: 2
layer_0/add?
layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice_2/stack?
layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_2/stack_1?
layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_2/stack_2?
layer_0/strided_slice_2StridedSlicelayer_0/Shape:output:0&layer_0/strided_slice_2/stack:output:0(layer_0/strided_slice_2/stack_1:output:0(layer_0/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice_2d
layer_0/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_0/mul_1/y?
layer_0/mul_1Mul layer_0/strided_slice_2:output:0layer_0/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_0/mul_1d
layer_0/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_0/add_1/yu
layer_0/add_1AddV2layer_0/mul_1:z:0layer_0/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_0/add_1?
&layer_0/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_0/conv2d_transpose/input_sizes/3?
$layer_0/conv2d_transpose/input_sizesPacklayer_0/strided_slice:output:0layer_0/add:z:0layer_0/add_1:z:0/layer_0/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_0/conv2d_transpose/input_sizes?
layer_0/conv2d_transposeConv2DBackpropInput-layer_0/conv2d_transpose/input_sizes:output:0layer_0/transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_0/conv2d_transpose?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAdd!layer_0/conv2d_transpose:output:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/BiasAddi
layer_0/igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/x?
layer_0/igdn_0/EqualEquallayer_0_igdn_0_equal_xlayer_0/igdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/Equal?
layer_0/igdn_0/condStatelessIflayer_0/igdn_0/Equal:z:0layer_0/igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_0_igdn_0_cond_false_203705*
output_shapes
: *2
then_branch#R!
layer_0_igdn_0_cond_true_2037042
layer_0/igdn_0/cond?
layer_0/igdn_0/cond/IdentityIdentitylayer_0/igdn_0/cond:output:0*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity?
layer_0/igdn_0/cond_1StatelessIf%layer_0/igdn_0/cond/Identity:output:0layer_0/BiasAdd:output:0layer_0_igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_0_igdn_0_cond_1_false_203716*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_1_true_2037152
layer_0/igdn_0/cond_1?
layer_0/igdn_0/cond_1/IdentityIdentitylayer_0/igdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203761*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203771*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
layer_0/igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/igdn_0/Reshape/shape?
layer_0/igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0%layer_0/igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/igdn_0/Reshape?
layer_0/igdn_0/convolutionConv2D'layer_0/igdn_0/cond_1/Identity:output:0layer_0/igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_0/igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203785*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
layer_0/igdn_0/BiasAddBiasAdd#layer_0/igdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/igdn_0/BiasAddm
layer_0/igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/x_1?
layer_0/igdn_0/Equal_1Equallayer_0_igdn_0_equal_1_xlayer_0/igdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/Equal_1?
layer_0/igdn_0/cond_2StatelessIflayer_0/igdn_0/Equal_1:z:0layer_0/igdn_0/BiasAdd:output:0layer_0_igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_0_igdn_0_cond_2_false_203799*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_2_true_2037982
layer_0/igdn_0/cond_2?
layer_0/igdn_0/cond_2/IdentityIdentitylayer_0/igdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity?
layer_0/igdn_0/mulMullayer_0/BiasAdd:output:0'layer_0/igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/igdn_0/mul?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshape?
layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_1/transpose/perm?
layer_1/transpose	Transposelayer_1/kernel/Reshape:output:0layer_1/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_1/transposed
layer_1/ShapeShapelayer_0/igdn_0/mul:z:0*
T0*
_output_shapes
:2
layer_1/Shape?
layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_1/strided_slice/stack?
layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice/stack_1?
layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice/stack_2?
layer_1/strided_sliceStridedSlicelayer_1/Shape:output:0$layer_1/strided_slice/stack:output:0&layer_1/strided_slice/stack_1:output:0&layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice?
layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice_1/stack?
layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_1/stack_1?
layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_1/stack_2?
layer_1/strided_slice_1StridedSlicelayer_1/Shape:output:0&layer_1/strided_slice_1/stack:output:0(layer_1/strided_slice_1/stack_1:output:0(layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice_1`
layer_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_1/mul/y|
layer_1/mulMul layer_1/strided_slice_1:output:0layer_1/mul/y:output:0*
T0*
_output_shapes
: 2
layer_1/mul`
layer_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_1/add/ym
layer_1/addAddV2layer_1/mul:z:0layer_1/add/y:output:0*
T0*
_output_shapes
: 2
layer_1/add?
layer_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice_2/stack?
layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_2/stack_1?
layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_2/stack_2?
layer_1/strided_slice_2StridedSlicelayer_1/Shape:output:0&layer_1/strided_slice_2/stack:output:0(layer_1/strided_slice_2/stack_1:output:0(layer_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice_2d
layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_1/mul_1/y?
layer_1/mul_1Mul layer_1/strided_slice_2:output:0layer_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_1/mul_1d
layer_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_1/add_1/yu
layer_1/add_1AddV2layer_1/mul_1:z:0layer_1/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_1/add_1?
&layer_1/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_1/conv2d_transpose/input_sizes/3?
$layer_1/conv2d_transpose/input_sizesPacklayer_1/strided_slice:output:0layer_1/add:z:0layer_1/add_1:z:0/layer_1/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_1/conv2d_transpose/input_sizes?
layer_1/conv2d_transposeConv2DBackpropInput-layer_1/conv2d_transpose/input_sizes:output:0layer_1/transpose:y:0layer_0/igdn_0/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_1/conv2d_transpose?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAdd!layer_1/conv2d_transpose:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/BiasAddi
layer_1/igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/x?
layer_1/igdn_1/EqualEquallayer_1_igdn_1_equal_xlayer_1/igdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/Equal?
layer_1/igdn_1/condStatelessIflayer_1/igdn_1/Equal:z:0layer_1/igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_1_igdn_1_cond_false_203866*
output_shapes
: *2
then_branch#R!
layer_1_igdn_1_cond_true_2038652
layer_1/igdn_1/cond?
layer_1/igdn_1/cond/IdentityIdentitylayer_1/igdn_1/cond:output:0*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity?
layer_1/igdn_1/cond_1StatelessIf%layer_1/igdn_1/cond/Identity:output:0layer_1/BiasAdd:output:0layer_1_igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_1_igdn_1_cond_1_false_203877*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_1_true_2038762
layer_1/igdn_1/cond_1?
layer_1/igdn_1/cond_1/IdentityIdentitylayer_1/igdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203922*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203932*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
layer_1/igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/igdn_1/Reshape/shape?
layer_1/igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0%layer_1/igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/igdn_1/Reshape?
layer_1/igdn_1/convolutionConv2D'layer_1/igdn_1/cond_1/Identity:output:0layer_1/igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_1/igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203946*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
layer_1/igdn_1/BiasAddBiasAdd#layer_1/igdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/igdn_1/BiasAddm
layer_1/igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/x_1?
layer_1/igdn_1/Equal_1Equallayer_1_igdn_1_equal_1_xlayer_1/igdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/Equal_1?
layer_1/igdn_1/cond_2StatelessIflayer_1/igdn_1/Equal_1:z:0layer_1/igdn_1/BiasAdd:output:0layer_1_igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_1_igdn_1_cond_2_false_203960*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_2_true_2039592
layer_1/igdn_1/cond_2?
layer_1/igdn_1/cond_2/IdentityIdentitylayer_1/igdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity?
layer_1/igdn_1/mulMullayer_1/BiasAdd:output:0'layer_1/igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/igdn_1/mul?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshape?
layer_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_2/transpose/perm?
layer_2/transpose	Transposelayer_2/kernel/Reshape:output:0layer_2/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_2/transposed
layer_2/ShapeShapelayer_1/igdn_1/mul:z:0*
T0*
_output_shapes
:2
layer_2/Shape?
layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_2/strided_slice/stack?
layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice/stack_1?
layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice/stack_2?
layer_2/strided_sliceStridedSlicelayer_2/Shape:output:0$layer_2/strided_slice/stack:output:0&layer_2/strided_slice/stack_1:output:0&layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice?
layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice_1/stack?
layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_1/stack_1?
layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_1/stack_2?
layer_2/strided_slice_1StridedSlicelayer_2/Shape:output:0&layer_2/strided_slice_1/stack:output:0(layer_2/strided_slice_1/stack_1:output:0(layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice_1`
layer_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_2/mul/y|
layer_2/mulMul layer_2/strided_slice_1:output:0layer_2/mul/y:output:0*
T0*
_output_shapes
: 2
layer_2/mul`
layer_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_2/add/ym
layer_2/addAddV2layer_2/mul:z:0layer_2/add/y:output:0*
T0*
_output_shapes
: 2
layer_2/add?
layer_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice_2/stack?
layer_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_2/stack_1?
layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_2/stack_2?
layer_2/strided_slice_2StridedSlicelayer_2/Shape:output:0&layer_2/strided_slice_2/stack:output:0(layer_2/strided_slice_2/stack_1:output:0(layer_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice_2d
layer_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_2/mul_1/y?
layer_2/mul_1Mul layer_2/strided_slice_2:output:0layer_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_2/mul_1d
layer_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_2/add_1/yu
layer_2/add_1AddV2layer_2/mul_1:z:0layer_2/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_2/add_1?
&layer_2/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_2/conv2d_transpose/input_sizes/3?
$layer_2/conv2d_transpose/input_sizesPacklayer_2/strided_slice:output:0layer_2/add:z:0layer_2/add_1:z:0/layer_2/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_2/conv2d_transpose/input_sizes?
layer_2/conv2d_transposeConv2DBackpropInput-layer_2/conv2d_transpose/input_sizes:output:0layer_2/transpose:y:0layer_1/igdn_1/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_2/conv2d_transpose?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAdd!layer_2/conv2d_transpose:output:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/BiasAddi
layer_2/igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/x?
layer_2/igdn_2/EqualEquallayer_2_igdn_2_equal_xlayer_2/igdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/Equal?
layer_2/igdn_2/condStatelessIflayer_2/igdn_2/Equal:z:0layer_2/igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_2_igdn_2_cond_false_204027*
output_shapes
: *2
then_branch#R!
layer_2_igdn_2_cond_true_2040262
layer_2/igdn_2/cond?
layer_2/igdn_2/cond/IdentityIdentitylayer_2/igdn_2/cond:output:0*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity?
layer_2/igdn_2/cond_1StatelessIf%layer_2/igdn_2/cond/Identity:output:0layer_2/BiasAdd:output:0layer_2_igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_2_igdn_2_cond_1_false_204038*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_1_true_2040372
layer_2/igdn_2/cond_1?
layer_2/igdn_2/cond_1/IdentityIdentitylayer_2/igdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204083*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204093*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
layer_2/igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/igdn_2/Reshape/shape?
layer_2/igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0%layer_2/igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/igdn_2/Reshape?
layer_2/igdn_2/convolutionConv2D'layer_2/igdn_2/cond_1/Identity:output:0layer_2/igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_2/igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204107*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
layer_2/igdn_2/BiasAddBiasAdd#layer_2/igdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/igdn_2/BiasAddm
layer_2/igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/x_1?
layer_2/igdn_2/Equal_1Equallayer_2_igdn_2_equal_1_xlayer_2/igdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/Equal_1?
layer_2/igdn_2/cond_2StatelessIflayer_2/igdn_2/Equal_1:z:0layer_2/igdn_2/BiasAdd:output:0layer_2_igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_2_igdn_2_cond_2_false_204121*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_2_true_2041202
layer_2/igdn_2/cond_2?
layer_2/igdn_2/cond_2/IdentityIdentitylayer_2/igdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity?
layer_2/igdn_2/mulMullayer_2/BiasAdd:output:0'layer_2/igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/igdn_2/mul?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshape?
layer_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_3/transpose/perm?
layer_3/transpose	Transposelayer_3/kernel/Reshape:output:0layer_3/transpose/perm:output:0*
T0*'
_output_shapes
:?2
layer_3/transposed
layer_3/ShapeShapelayer_2/igdn_2/mul:z:0*
T0*
_output_shapes
:2
layer_3/Shape?
layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_3/strided_slice/stack?
layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice/stack_1?
layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice/stack_2?
layer_3/strided_sliceStridedSlicelayer_3/Shape:output:0$layer_3/strided_slice/stack:output:0&layer_3/strided_slice/stack_1:output:0&layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice?
layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice_1/stack?
layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_1/stack_1?
layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_1/stack_2?
layer_3/strided_slice_1StridedSlicelayer_3/Shape:output:0&layer_3/strided_slice_1/stack:output:0(layer_3/strided_slice_1/stack_1:output:0(layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice_1`
layer_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_3/mul/y|
layer_3/mulMul layer_3/strided_slice_1:output:0layer_3/mul/y:output:0*
T0*
_output_shapes
: 2
layer_3/mul`
layer_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_3/add/ym
layer_3/addAddV2layer_3/mul:z:0layer_3/add/y:output:0*
T0*
_output_shapes
: 2
layer_3/add?
layer_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice_2/stack?
layer_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_2/stack_1?
layer_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_2/stack_2?
layer_3/strided_slice_2StridedSlicelayer_3/Shape:output:0&layer_3/strided_slice_2/stack:output:0(layer_3/strided_slice_2/stack_1:output:0(layer_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice_2d
layer_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_3/mul_1/y?
layer_3/mul_1Mul layer_3/strided_slice_2:output:0layer_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_3/mul_1d
layer_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_3/add_1/yu
layer_3/add_1AddV2layer_3/mul_1:z:0layer_3/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_3/add_1?
&layer_3/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&layer_3/conv2d_transpose/input_sizes/3?
$layer_3/conv2d_transpose/input_sizesPacklayer_3/strided_slice:output:0layer_3/add:z:0layer_3/add_1:z:0/layer_3/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_3/conv2d_transpose/input_sizes?
layer_3/conv2d_transposeConv2DBackpropInput-layer_3/conv2d_transpose/input_sizes:output:0layer_3/transpose:y:0layer_2/igdn_2/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_3/conv2d_transpose?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAdd!layer_3/conv2d_transpose:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2
layer_3/BiasAdde
lambda_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
lambda_1/mul/y?
lambda_1/mulMullayer_3/BiasAdd:output:0lambda_1/mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_1/mul?
IdentityIdentitylambda_1/mul:z:0^layer_0/BiasAdd/ReadVariableOp/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2@
layer_0/BiasAdd/ReadVariableOplayer_0/BiasAdd/ReadVariableOp2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
&layer_1_igdn_1_cond_2_cond_true_203444:
6layer_1_igdn_1_cond_2_cond_sqrt_layer_1_igdn_1_biasadd*
&layer_1_igdn_1_cond_2_cond_placeholder'
#layer_1_igdn_1_cond_2_cond_identity?
layer_1/igdn_1/cond_2/cond/SqrtSqrt6layer_1_igdn_1_cond_2_cond_sqrt_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_1/igdn_1/cond_2/cond/Sqrt?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity#layer_1/igdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity"S
#layer_1_igdn_1_cond_2_cond_identity,layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0model_synthesis_layer_1_igdn_1_cond_false_200196U
Qmodel_synthesis_layer_1_igdn_1_cond_identity_model_synthesis_layer_1_igdn_1_equal
0
,model_synthesis_layer_1_igdn_1_cond_identity
?
,model/synthesis/layer_1/igdn_1/cond/IdentityIdentityQmodel_synthesis_layer_1_igdn_1_cond_identity_model_synthesis_layer_1_igdn_1_equal*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_1/igdn_1/cond/Identity"e
,model_synthesis_layer_1_igdn_1_cond_identity5model/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?*
?
C__inference_layer_3_layer_call_and_return_conditional_losses_204737

inputs
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_3/kernel/Reshape:output:0transpose/perm:output:0*
T0*'
_output_shapes
:?2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:,????????????????????????????:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

:
?
?
igdn_2_cond_1_cond_false_204591#
igdn_2_cond_1_cond_cond_biasadd
igdn_2_cond_1_cond_equal_x
igdn_2_cond_1_cond_identityq
igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_2/cond_1/cond/x?
igdn_2/cond_1/cond/EqualEqualigdn_2_cond_1_cond_equal_xigdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_1/cond/Equal?
igdn_2/cond_1/cond/condStatelessIfigdn_2/cond_1/cond/Equal:z:0igdn_2_cond_1_cond_cond_biasaddigdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_2_cond_1_cond_cond_false_204601*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_2_cond_1_cond_cond_true_2046002
igdn_2/cond_1/cond/cond?
 igdn_2/cond_1/cond/cond/IdentityIdentity igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity?
igdn_2/cond_1/cond/IdentityIdentity)igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity"C
igdn_2_cond_1_cond_identity$igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_2_cond_2_false_201035%
!igdn_2_cond_2_cond_igdn_2_biasadd
igdn_2_cond_2_equal_x
igdn_2_cond_2_identityg
igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_2/cond_2/x?
igdn_2/cond_2/EqualEqualigdn_2_cond_2_equal_xigdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_2/Equal?
igdn_2/cond_2/condStatelessIfigdn_2/cond_2/Equal:z:0!igdn_2_cond_2_cond_igdn_2_biasaddigdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_2_cond_2_cond_false_201044*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_2_cond_true_2010432
igdn_2/cond_2/cond?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity?
igdn_2/cond_2/IdentityIdentity$igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity"9
igdn_2_cond_2_identityigdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_2_true_200845)
%igdn_1_cond_2_identity_igdn_1_biasadd
igdn_1_cond_2_placeholder
igdn_1_cond_2_identity?
igdn_1/cond_2/IdentityIdentity%igdn_1_cond_2_identity_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity"9
igdn_1_cond_2_identityigdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_2_cond_1_cond_cond_true_204600*
&igdn_2_cond_1_cond_cond_square_biasadd'
#igdn_2_cond_1_cond_cond_placeholder$
 igdn_2_cond_1_cond_cond_identity?
igdn_2/cond_1/cond/cond/SquareSquare&igdn_2_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_2/cond_1/cond/cond/Square?
 igdn_2/cond_1/cond/cond/IdentityIdentity"igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity"M
 igdn_2_cond_1_cond_cond_identity)igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_0_igdn_0_cond_1_cond_false_202153G
Csynthesis_layer_0_igdn_0_cond_1_cond_cond_synthesis_layer_0_biasadd0
,synthesis_layer_0_igdn_0_cond_1_cond_equal_x1
-synthesis_layer_0_igdn_0_cond_1_cond_identity?
&synthesis/layer_0/igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_0/igdn_0/cond_1/cond/x?
*synthesis/layer_0/igdn_0/cond_1/cond/EqualEqual,synthesis_layer_0_igdn_0_cond_1_cond_equal_x/synthesis/layer_0/igdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_0/igdn_0/cond_1/cond/Equal?
)synthesis/layer_0/igdn_0/cond_1/cond/condStatelessIf.synthesis/layer_0/igdn_0/cond_1/cond/Equal:z:0Csynthesis_layer_0_igdn_0_cond_1_cond_cond_synthesis_layer_0_biasadd,synthesis_layer_0_igdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202163*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_2021622+
)synthesis/layer_0/igdn_0/cond_1/cond/cond?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity2synthesis/layer_0/igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_1_cond_identity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_201783

inputs
synthesis_201709$
synthesis_201711:
??
synthesis_201713:	?
synthesis_201715$
synthesis_201717:
??
synthesis_201719
synthesis_201721
synthesis_201723:	?
synthesis_201725
synthesis_201727
synthesis_201729
synthesis_201731$
synthesis_201733:
??
synthesis_201735:	?
synthesis_201737$
synthesis_201739:
??
synthesis_201741
synthesis_201743
synthesis_201745:	?
synthesis_201747
synthesis_201749
synthesis_201751
synthesis_201753$
synthesis_201755:
??
synthesis_201757:	?
synthesis_201759$
synthesis_201761:
??
synthesis_201763
synthesis_201765
synthesis_201767:	?
synthesis_201769
synthesis_201771
synthesis_201773
synthesis_201775#
synthesis_201777:	?
synthesis_201779:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinputssynthesis_201709synthesis_201711synthesis_201713synthesis_201715synthesis_201717synthesis_201719synthesis_201721synthesis_201723synthesis_201725synthesis_201727synthesis_201729synthesis_201731synthesis_201733synthesis_201735synthesis_201737synthesis_201739synthesis_201741synthesis_201743synthesis_201745synthesis_201747synthesis_201749synthesis_201751synthesis_201753synthesis_201755synthesis_201757synthesis_201759synthesis_201761synthesis_201763synthesis_201765synthesis_201767synthesis_201769synthesis_201771synthesis_201773synthesis_201775synthesis_201777synthesis_201779*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2013152#
!synthesis/StatefulPartitionedCall?
IdentityIdentity*synthesis/StatefulPartitionedCall:output:0"^synthesis/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2F
!synthesis/StatefulPartitionedCall!synthesis/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_201228

inputs
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
mul/yu
mulMulinputsmul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
0synthesis_layer_0_igdn_0_cond_2_cond_true_202759N
Jsynthesis_layer_0_igdn_0_cond_2_cond_sqrt_synthesis_layer_0_igdn_0_biasadd4
0synthesis_layer_0_igdn_0_cond_2_cond_placeholder1
-synthesis_layer_0_igdn_0_cond_2_cond_identity?
)synthesis/layer_0/igdn_0/cond_2/cond/SqrtSqrtJsynthesis_layer_0_igdn_0_cond_2_cond_sqrt_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_0/igdn_0/cond_2/cond/Sqrt?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_2_cond_identity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201315

inputs
layer_0_201237"
layer_0_201239:
??
layer_0_201241:	?
layer_0_201243"
layer_0_201245:
??
layer_0_201247
layer_0_201249
layer_0_201251:	?
layer_0_201253
layer_0_201255
layer_0_201257
layer_1_201260"
layer_1_201262:
??
layer_1_201264:	?
layer_1_201266"
layer_1_201268:
??
layer_1_201270
layer_1_201272
layer_1_201274:	?
layer_1_201276
layer_1_201278
layer_1_201280
layer_2_201283"
layer_2_201285:
??
layer_2_201287:	?
layer_2_201289"
layer_2_201291:
??
layer_2_201293
layer_2_201295
layer_2_201297:	?
layer_2_201299
layer_2_201301
layer_2_201303
layer_3_201306!
layer_3_201308:	?
layer_3_201310:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_201237layer_0_201239layer_0_201241layer_0_201243layer_0_201245layer_0_201247layer_0_201249layer_0_201251layer_0_201253layer_0_201255layer_0_201257*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_2006862!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201260layer_1_201262layer_1_201264layer_1_201266layer_1_201268layer_1_201270layer_1_201272layer_1_201274layer_1_201276layer_1_201278layer_1_201280*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_1_layer_call_and_return_conditional_losses_2008752!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201283layer_2_201285layer_2_201287layer_2_201289layer_2_201291layer_2_201293layer_2_201295layer_2_201297layer_2_201299layer_2_201301layer_2_201303*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_2_layer_call_and_return_conditional_losses_2010642!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201306layer_3_201308layer_3_201310*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_2011272!
layer_3/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011412
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
*synthesis_layer_2_igdn_2_cond_false_202979I
Esynthesis_layer_2_igdn_2_cond_identity_synthesis_layer_2_igdn_2_equal
*
&synthesis_layer_2_igdn_2_cond_identity
?
&synthesis/layer_2/igdn_2/cond/IdentityIdentityEsynthesis_layer_2_igdn_2_cond_identity_synthesis_layer_2_igdn_2_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity"Y
&synthesis_layer_2_igdn_2_cond_identity/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?	
?
;model_synthesis_layer_2_igdn_2_cond_1_cond_cond_true_200386Z
Vmodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_square_model_synthesis_layer_2_biasadd?
;model_synthesis_layer_2_igdn_2_cond_1_cond_cond_placeholder<
8model_synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
6model/synthesis/layer_2/igdn_2/cond_1/cond/cond/SquareSquareVmodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_square_model_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6model/synthesis/layer_2/igdn_2/cond_1/cond/cond/Square?
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity:model/synthesis/layer_2/igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"}
8model_synthesis_layer_2_igdn_2_cond_1_cond_cond_identityAmodel/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+layer_1_igdn_1_cond_1_cond_cond_true_203371:
6layer_1_igdn_1_cond_1_cond_cond_square_layer_1_biasadd/
+layer_1_igdn_1_cond_1_cond_cond_placeholder,
(layer_1_igdn_1_cond_1_cond_cond_identity?
&layer_1/igdn_1/cond_1/cond/cond/SquareSquare6layer_1_igdn_1_cond_1_cond_cond_square_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_1/igdn_1/cond_1/cond/cond/Square?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity*layer_1/igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity"]
(layer_1_igdn_1_cond_1_cond_cond_identity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
A__inference_model_layer_call_and_return_conditional_losses_201703
input_2
synthesis_201629$
synthesis_201631:
??
synthesis_201633:	?
synthesis_201635$
synthesis_201637:
??
synthesis_201639
synthesis_201641
synthesis_201643:	?
synthesis_201645
synthesis_201647
synthesis_201649
synthesis_201651$
synthesis_201653:
??
synthesis_201655:	?
synthesis_201657$
synthesis_201659:
??
synthesis_201661
synthesis_201663
synthesis_201665:	?
synthesis_201667
synthesis_201669
synthesis_201671
synthesis_201673$
synthesis_201675:
??
synthesis_201677:	?
synthesis_201679$
synthesis_201681:
??
synthesis_201683
synthesis_201685
synthesis_201687:	?
synthesis_201689
synthesis_201691
synthesis_201693
synthesis_201695#
synthesis_201697:	?
synthesis_201699:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinput_2synthesis_201629synthesis_201631synthesis_201633synthesis_201635synthesis_201637synthesis_201639synthesis_201641synthesis_201643synthesis_201645synthesis_201647synthesis_201649synthesis_201651synthesis_201653synthesis_201655synthesis_201657synthesis_201659synthesis_201661synthesis_201663synthesis_201665synthesis_201667synthesis_201669synthesis_201671synthesis_201673synthesis_201675synthesis_201677synthesis_201679synthesis_201681synthesis_201683synthesis_201685synthesis_201687synthesis_201689synthesis_201691synthesis_201693synthesis_201695synthesis_201697synthesis_201699*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2014732#
!synthesis/StatefulPartitionedCall?
IdentityIdentity*synthesis/StatefulPartitionedCall:output:0"^synthesis/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2F
!synthesis/StatefulPartitionedCall!synthesis/StatefulPartitionedCall:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
igdn_0_cond_2_cond_false_204336)
%igdn_0_cond_2_cond_pow_igdn_0_biasadd
igdn_0_cond_2_cond_pow_y
igdn_0_cond_2_cond_identity?
igdn_0/cond_2/cond/powPow%igdn_0_cond_2_cond_pow_igdn_0_biasaddigdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/pow?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity"C
igdn_0_cond_2_cond_identity$igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_0_cond_false_200563%
!igdn_0_cond_identity_igdn_0_equal

igdn_0_cond_identity
|
igdn_0/cond/IdentityIdentity!igdn_0_cond_identity_igdn_0_equal*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity"5
igdn_0_cond_identityigdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
*synthesis_layer_0_igdn_0_cond_false_202133I
Esynthesis_layer_0_igdn_0_cond_identity_synthesis_layer_0_igdn_0_equal
*
&synthesis_layer_0_igdn_0_cond_identity
?
&synthesis/layer_0/igdn_0/cond/IdentityIdentityEsynthesis_layer_0_igdn_0_cond_identity_synthesis_layer_0_igdn_0_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity"Y
&synthesis_layer_0_igdn_0_cond_identity/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_1_igdn_1_cond_1_true_202828F
Bsynthesis_layer_1_igdn_1_cond_1_identity_synthesis_layer_1_biasadd/
+synthesis_layer_1_igdn_1_cond_1_placeholder,
(synthesis_layer_1_igdn_1_cond_1_identity?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentityBsynthesis_layer_1_igdn_1_cond_1_identity_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity"]
(synthesis_layer_1_igdn_1_cond_1_identity1synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_2_igdn_2_cond_2_cond_true_203081N
Jsynthesis_layer_2_igdn_2_cond_2_cond_sqrt_synthesis_layer_2_igdn_2_biasadd4
0synthesis_layer_2_igdn_2_cond_2_cond_placeholder1
-synthesis_layer_2_igdn_2_cond_2_cond_identity?
)synthesis/layer_2/igdn_2/cond_2/cond/SqrtSqrtJsynthesis_layer_2_igdn_2_cond_2_cond_sqrt_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_2/igdn_2/cond_2/cond/Sqrt?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_2_cond_identity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_0_igdn_0_cond_1_cond_true_2032002
.layer_0_igdn_0_cond_1_cond_abs_layer_0_biasadd*
&layer_0_igdn_0_cond_1_cond_placeholder'
#layer_0_igdn_0_cond_1_cond_identity?
layer_0/igdn_0/cond_1/cond/AbsAbs.layer_0_igdn_0_cond_1_cond_abs_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/cond/Abs?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity"layer_0/igdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity"S
#layer_0_igdn_0_cond_1_cond_identity,layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,layer_2_igdn_2_cond_1_cond_cond_false_2040577
3layer_2_igdn_2_cond_1_cond_cond_pow_layer_2_biasadd)
%layer_2_igdn_2_cond_1_cond_cond_pow_y,
(layer_2_igdn_2_cond_1_cond_cond_identity?
#layer_2/igdn_2/cond_1/cond/cond/powPow3layer_2_igdn_2_cond_1_cond_cond_pow_layer_2_biasadd%layer_2_igdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/cond/pow?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity'layer_2/igdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity"]
(layer_2_igdn_2_cond_1_cond_cond_identity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_2_igdn_2_cond_2_cond_true_204129:
6layer_2_igdn_2_cond_2_cond_sqrt_layer_2_igdn_2_biasadd*
&layer_2_igdn_2_cond_2_cond_placeholder'
#layer_2_igdn_2_cond_2_cond_identity?
layer_2/igdn_2/cond_2/cond/SqrtSqrt6layer_2_igdn_2_cond_2_cond_sqrt_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_2/igdn_2/cond_2/cond/Sqrt?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity#layer_2/igdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity"S
#layer_2_igdn_2_cond_2_cond_identity,layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
/model_synthesis_layer_0_igdn_0_cond_true_2000343
/model_synthesis_layer_0_igdn_0_cond_placeholder
0
,model_synthesis_layer_0_igdn_0_cond_identity
?
)model/synthesis/layer_0/igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/synthesis/layer_0/igdn_0/cond/Const?
,model/synthesis/layer_0/igdn_0/cond/IdentityIdentity2model/synthesis/layer_0/igdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_0/igdn_0/cond/Identity"e
,model_synthesis_layer_0_igdn_0_cond_identity5model/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
igdn_0_cond_1_cond_false_200583#
igdn_0_cond_1_cond_cond_biasadd
igdn_0_cond_1_cond_equal_x
igdn_0_cond_1_cond_identityq
igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_0/cond_1/cond/x?
igdn_0/cond_1/cond/EqualEqualigdn_0_cond_1_cond_equal_xigdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_1/cond/Equal?
igdn_0/cond_1/cond/condStatelessIfigdn_0/cond_1/cond/Equal:z:0igdn_0_cond_1_cond_cond_biasaddigdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_0_cond_1_cond_cond_false_200593*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_0_cond_1_cond_cond_true_2005922
igdn_0/cond_1/cond/cond?
 igdn_0/cond_1/cond/cond/IdentityIdentity igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity?
igdn_0/cond_1/cond/IdentityIdentity)igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity"C
igdn_0_cond_1_cond_identity$igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
)synthesis_layer_0_igdn_0_cond_true_202656-
)synthesis_layer_0_igdn_0_cond_placeholder
*
&synthesis_layer_0_igdn_0_cond_identity
?
#synthesis/layer_0/igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_0/igdn_0/cond/Const?
&synthesis/layer_0/igdn_0/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity"Y
&synthesis_layer_0_igdn_0_cond_identity/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1synthesis_layer_2_igdn_2_cond_2_cond_false_203082M
Isynthesis_layer_2_igdn_2_cond_2_cond_pow_synthesis_layer_2_igdn_2_biasadd.
*synthesis_layer_2_igdn_2_cond_2_cond_pow_y1
-synthesis_layer_2_igdn_2_cond_2_cond_identity?
(synthesis/layer_2/igdn_2/cond_2/cond/powPowIsynthesis_layer_2_igdn_2_cond_2_cond_pow_synthesis_layer_2_igdn_2_biasadd*synthesis_layer_2_igdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/cond/pow?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_2_cond_identity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?*
?
C__inference_layer_3_layer_call_and_return_conditional_losses_201127

inputs
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_3/kernel/Reshape:output:0transpose/perm:output:0*
T0*'
_output_shapes
:?2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*O
_input_shapes>
<:,????????????????????????????:: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

:
?
?
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_202686N
Jsynthesis_layer_0_igdn_0_cond_1_cond_cond_square_synthesis_layer_0_biasadd9
5synthesis_layer_0_igdn_0_cond_1_cond_cond_placeholder6
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
0synthesis/layer_0/igdn_0/cond_1/cond/cond/SquareSquareJsynthesis_layer_0_igdn_0_cond_1_cond_cond_square_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_0/igdn_0/cond_1/cond/cond/Square?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity4synthesis/layer_0/igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"q
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_2_cond_1_true_204581"
igdn_2_cond_1_identity_biasadd
igdn_2_cond_1_placeholder
igdn_2_cond_1_identity?
igdn_2/cond_1/IdentityIdentityigdn_2_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity"9
igdn_2_cond_1_identityigdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_1_igdn_1_cond_2_cond_false_202921M
Isynthesis_layer_1_igdn_1_cond_2_cond_pow_synthesis_layer_1_igdn_1_biasadd.
*synthesis_layer_1_igdn_1_cond_2_cond_pow_y1
-synthesis_layer_1_igdn_1_cond_2_cond_identity?
(synthesis/layer_1/igdn_1/cond_2/cond/powPowIsynthesis_layer_1_igdn_1_cond_2_cond_pow_synthesis_layer_1_igdn_1_biasadd*synthesis_layer_1_igdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/cond/pow?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_2_cond_identity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_0_igdn_0_cond_2_cond_false_202236M
Isynthesis_layer_0_igdn_0_cond_2_cond_pow_synthesis_layer_0_igdn_0_biasadd.
*synthesis_layer_0_igdn_0_cond_2_cond_pow_y1
-synthesis_layer_0_igdn_0_cond_2_cond_identity?
(synthesis/layer_0/igdn_0/cond_2/cond/powPowIsynthesis_layer_0_igdn_0_cond_2_cond_pow_synthesis_layer_0_igdn_0_biasadd*synthesis_layer_0_igdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/cond/pow?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_2_cond_identity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_1_igdn_1_cond_1_false_203877.
*layer_1_igdn_1_cond_1_cond_layer_1_biasadd!
layer_1_igdn_1_cond_1_equal_x"
layer_1_igdn_1_cond_1_identityw
layer_1/igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/cond_1/x?
layer_1/igdn_1/cond_1/EqualEquallayer_1_igdn_1_cond_1_equal_x layer_1/igdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/cond_1/Equal?
layer_1/igdn_1/cond_1/condStatelessIflayer_1/igdn_1/cond_1/Equal:z:0*layer_1_igdn_1_cond_1_cond_layer_1_biasaddlayer_1_igdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_1_igdn_1_cond_1_cond_false_203886*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_1_cond_true_2038852
layer_1/igdn_1/cond_1/cond?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity#layer_1/igdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity?
layer_1/igdn_1/cond_1/IdentityIdentity,layer_1/igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity"I
layer_1_igdn_1_cond_1_identity'layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_0_igdn_0_cond_2_false_202751I
Esynthesis_layer_0_igdn_0_cond_2_cond_synthesis_layer_0_igdn_0_biasadd+
'synthesis_layer_0_igdn_0_cond_2_equal_x,
(synthesis_layer_0_igdn_0_cond_2_identity?
!synthesis/layer_0/igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_0/igdn_0/cond_2/x?
%synthesis/layer_0/igdn_0/cond_2/EqualEqual'synthesis_layer_0_igdn_0_cond_2_equal_x*synthesis/layer_0/igdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_0/igdn_0/cond_2/Equal?
$synthesis/layer_0/igdn_0/cond_2/condStatelessIf)synthesis/layer_0/igdn_0/cond_2/Equal:z:0Esynthesis_layer_0_igdn_0_cond_2_cond_synthesis_layer_0_igdn_0_biasadd'synthesis_layer_0_igdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_0_igdn_0_cond_2_cond_false_202760*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_2_cond_true_2027592&
$synthesis/layer_0/igdn_0/cond_2/cond?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity"]
(synthesis_layer_0_igdn_0_cond_2_identity1synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_0_igdn_0_cond_false_2031815
1layer_0_igdn_0_cond_identity_layer_0_igdn_0_equal
 
layer_0_igdn_0_cond_identity
?
layer_0/igdn_0/cond/IdentityIdentity1layer_0_igdn_0_cond_identity_layer_0_igdn_0_equal*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity"E
layer_0_igdn_0_cond_identity%layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
P
igdn_2_cond_true_204570
igdn_2_cond_placeholder

igdn_2_cond_identity
h
igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_2/cond/Constu
igdn_2/cond/IdentityIdentityigdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity"5
igdn_2_cond_identityigdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_0_igdn_0_cond_1_true_202143F
Bsynthesis_layer_0_igdn_0_cond_1_identity_synthesis_layer_0_biasadd/
+synthesis_layer_0_igdn_0_cond_1_placeholder,
(synthesis_layer_0_igdn_0_cond_1_identity?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentityBsynthesis_layer_0_igdn_0_cond_1_identity_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity"]
(synthesis_layer_0_igdn_0_cond_1_identity1synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_202012
input_2
unknown
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3:
??
	unknown_4
	unknown_5
	unknown_6:	?
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:
??

unknown_12:	?

unknown_13

unknown_14:
??

unknown_15

unknown_16

unknown_17:	?

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22:
??

unknown_23:	?

unknown_24

unknown_25:
??

unknown_26

unknown_27

unknown_28:	?

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33:	?

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2019372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
'layer_2_igdn_2_cond_1_cond_false_2035233
/layer_2_igdn_2_cond_1_cond_cond_layer_2_biasadd&
"layer_2_igdn_2_cond_1_cond_equal_x'
#layer_2_igdn_2_cond_1_cond_identity?
layer_2/igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_2/igdn_2/cond_1/cond/x?
 layer_2/igdn_2/cond_1/cond/EqualEqual"layer_2_igdn_2_cond_1_cond_equal_x%layer_2/igdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_2/igdn_2/cond_1/cond/Equal?
layer_2/igdn_2/cond_1/cond/condStatelessIf$layer_2/igdn_2/cond_1/cond/Equal:z:0/layer_2_igdn_2_cond_1_cond_cond_layer_2_biasadd"layer_2_igdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_2_igdn_2_cond_1_cond_cond_false_203533*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_2_igdn_2_cond_1_cond_cond_true_2035322!
layer_2/igdn_2/cond_1/cond/cond?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity(layer_2/igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity"S
#layer_2_igdn_2_cond_1_cond_identity,layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_0_igdn_0_cond_1_cond_true_202676F
Bsynthesis_layer_0_igdn_0_cond_1_cond_abs_synthesis_layer_0_biasadd4
0synthesis_layer_0_igdn_0_cond_1_cond_placeholder1
-synthesis_layer_0_igdn_0_cond_1_cond_identity?
(synthesis/layer_0/igdn_0/cond_1/cond/AbsAbsBsynthesis_layer_0_igdn_0_cond_1_cond_abs_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/cond/Abs?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_1_cond_identity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$igdn_0_cond_1_cond_cond_false_204263'
#igdn_0_cond_1_cond_cond_pow_biasadd!
igdn_0_cond_1_cond_cond_pow_y$
 igdn_0_cond_1_cond_cond_identity?
igdn_0/cond_1/cond/cond/powPow#igdn_0_cond_1_cond_cond_pow_biasaddigdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/cond/pow?
 igdn_0/cond_1/cond/cond/IdentityIdentityigdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity"M
 igdn_0_cond_1_cond_cond_identity)igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_0_igdn_0_cond_2_false_2032755
1layer_0_igdn_0_cond_2_cond_layer_0_igdn_0_biasadd!
layer_0_igdn_0_cond_2_equal_x"
layer_0_igdn_0_cond_2_identityw
layer_0/igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_0/igdn_0/cond_2/x?
layer_0/igdn_0/cond_2/EqualEquallayer_0_igdn_0_cond_2_equal_x layer_0/igdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/cond_2/Equal?
layer_0/igdn_0/cond_2/condStatelessIflayer_0/igdn_0/cond_2/Equal:z:01layer_0_igdn_0_cond_2_cond_layer_0_igdn_0_biasaddlayer_0_igdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_0_igdn_0_cond_2_cond_false_203284*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_2_cond_true_2032832
layer_0/igdn_0/cond_2/cond?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity#layer_0/igdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity?
layer_0/igdn_0/cond_2/IdentityIdentity,layer_0/igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity"I
layer_0_igdn_0_cond_2_identity'layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_2_cond_1_cond_false_200961#
igdn_2_cond_1_cond_cond_biasadd
igdn_2_cond_1_cond_equal_x
igdn_2_cond_1_cond_identityq
igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_2/cond_1/cond/x?
igdn_2/cond_1/cond/EqualEqualigdn_2_cond_1_cond_equal_xigdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_1/cond/Equal?
igdn_2/cond_1/cond/condStatelessIfigdn_2/cond_1/cond/Equal:z:0igdn_2_cond_1_cond_cond_biasaddigdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_2_cond_1_cond_cond_false_200971*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_2_cond_1_cond_cond_true_2009702
igdn_2/cond_1/cond/cond?
 igdn_2/cond_1/cond/cond/IdentityIdentity igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity?
igdn_2/cond_1/cond/IdentityIdentity)igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity"C
igdn_2_cond_1_cond_identity$igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_2_igdn_2_cond_2_false_203073I
Esynthesis_layer_2_igdn_2_cond_2_cond_synthesis_layer_2_igdn_2_biasadd+
'synthesis_layer_2_igdn_2_cond_2_equal_x,
(synthesis_layer_2_igdn_2_cond_2_identity?
!synthesis/layer_2/igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_2/igdn_2/cond_2/x?
%synthesis/layer_2/igdn_2/cond_2/EqualEqual'synthesis_layer_2_igdn_2_cond_2_equal_x*synthesis/layer_2/igdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_2/igdn_2/cond_2/Equal?
$synthesis/layer_2/igdn_2/cond_2/condStatelessIf)synthesis/layer_2/igdn_2/cond_2/Equal:z:0Esynthesis_layer_2_igdn_2_cond_2_cond_synthesis_layer_2_igdn_2_biasadd'synthesis_layer_2_igdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_2_igdn_2_cond_2_cond_false_203082*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_2_cond_true_2030812&
$synthesis/layer_2/igdn_2/cond_2/cond?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity"]
(synthesis_layer_2_igdn_2_cond_2_identity1synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6model_synthesis_layer_0_igdn_0_cond_1_cond_true_200054R
Nmodel_synthesis_layer_0_igdn_0_cond_1_cond_abs_model_synthesis_layer_0_biasadd:
6model_synthesis_layer_0_igdn_0_cond_1_cond_placeholder7
3model_synthesis_layer_0_igdn_0_cond_1_cond_identity?
.model/synthesis/layer_0/igdn_0/cond_1/cond/AbsAbsNmodel_synthesis_layer_0_igdn_0_cond_1_cond_abs_model_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_1/cond/Abs?
3model/synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity2model/synthesis/layer_0/igdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_1/cond/Identity"s
3model_synthesis_layer_0_igdn_0_cond_1_cond_identity<model/synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_2_igdn_2_cond_2_cond_true_202557N
Jsynthesis_layer_2_igdn_2_cond_2_cond_sqrt_synthesis_layer_2_igdn_2_biasadd4
0synthesis_layer_2_igdn_2_cond_2_cond_placeholder1
-synthesis_layer_2_igdn_2_cond_2_cond_identity?
)synthesis/layer_2/igdn_2/cond_2/cond/SqrtSqrtJsynthesis_layer_2_igdn_2_cond_2_cond_sqrt_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_2/igdn_2/cond_2/cond/Sqrt?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_2_cond_identity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_0_cond_1_cond_cond_true_200592*
&igdn_0_cond_1_cond_cond_square_biasadd'
#igdn_0_cond_1_cond_cond_placeholder$
 igdn_0_cond_1_cond_cond_identity?
igdn_0/cond_1/cond/cond/SquareSquare&igdn_0_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_0/cond_1/cond/cond/Square?
 igdn_0/cond_1/cond/cond/IdentityIdentity"igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity"M
 igdn_0_cond_1_cond_cond_identity)igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_1_cond_true_204421"
igdn_1_cond_1_cond_abs_biasadd"
igdn_1_cond_1_cond_placeholder
igdn_1_cond_1_cond_identity?
igdn_1/cond_1/cond/AbsAbsigdn_1_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Abs?
igdn_1/cond_1/cond/IdentityIdentityigdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity"C
igdn_1_cond_1_cond_identity$igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_2_igdn_2_cond_false_2040275
1layer_2_igdn_2_cond_identity_layer_2_igdn_2_equal
 
layer_2_igdn_2_cond_identity
?
layer_2/igdn_2/cond/IdentityIdentity1layer_2_igdn_2_cond_identity_layer_2_igdn_2_equal*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity"E
layer_2_igdn_2_cond_identity%layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
igdn_1_cond_2_cond_true_204504*
&igdn_1_cond_2_cond_sqrt_igdn_1_biasadd"
igdn_1_cond_2_cond_placeholder
igdn_1_cond_2_cond_identity?
igdn_1/cond_2/cond/SqrtSqrt&igdn_1_cond_2_cond_sqrt_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Sqrt?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity"C
igdn_1_cond_2_cond_identity$igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
)synthesis_layer_1_igdn_1_cond_true_202293-
)synthesis_layer_1_igdn_1_cond_placeholder
*
&synthesis_layer_1_igdn_1_cond_identity
?
#synthesis/layer_1/igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_1/igdn_1/cond/Const?
&synthesis/layer_1/igdn_1/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity"Y
&synthesis_layer_1_igdn_1_cond_identity/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+layer_1_igdn_1_cond_1_cond_cond_true_203895:
6layer_1_igdn_1_cond_1_cond_cond_square_layer_1_biasadd/
+layer_1_igdn_1_cond_1_cond_cond_placeholder,
(layer_1_igdn_1_cond_1_cond_cond_identity?
&layer_1/igdn_1/cond_1/cond/cond/SquareSquare6layer_1_igdn_1_cond_1_cond_cond_square_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_1/igdn_1/cond_1/cond/cond/Square?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity*layer_1/igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity"]
(layer_1_igdn_1_cond_1_cond_cond_identity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_1_igdn_1_cond_2_false_202912I
Esynthesis_layer_1_igdn_1_cond_2_cond_synthesis_layer_1_igdn_1_biasadd+
'synthesis_layer_1_igdn_1_cond_2_equal_x,
(synthesis_layer_1_igdn_1_cond_2_identity?
!synthesis/layer_1/igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_1/igdn_1/cond_2/x?
%synthesis/layer_1/igdn_1/cond_2/EqualEqual'synthesis_layer_1_igdn_1_cond_2_equal_x*synthesis/layer_1/igdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_1/igdn_1/cond_2/Equal?
$synthesis/layer_1/igdn_1/cond_2/condStatelessIf)synthesis/layer_1/igdn_1/cond_2/Equal:z:0Esynthesis_layer_1_igdn_1_cond_2_cond_synthesis_layer_1_igdn_1_biasadd'synthesis_layer_1_igdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_1_igdn_1_cond_2_cond_false_202921*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_2_cond_true_2029202&
$synthesis/layer_1/igdn_1/cond_2/cond?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity"]
(synthesis_layer_1_igdn_1_cond_2_identity1synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_0_igdn_0_cond_1_cond_false_2037253
/layer_0_igdn_0_cond_1_cond_cond_layer_0_biasadd&
"layer_0_igdn_0_cond_1_cond_equal_x'
#layer_0_igdn_0_cond_1_cond_identity?
layer_0/igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_0/igdn_0/cond_1/cond/x?
 layer_0/igdn_0/cond_1/cond/EqualEqual"layer_0_igdn_0_cond_1_cond_equal_x%layer_0/igdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_0/igdn_0/cond_1/cond/Equal?
layer_0/igdn_0/cond_1/cond/condStatelessIf$layer_0/igdn_0/cond_1/cond/Equal:z:0/layer_0_igdn_0_cond_1_cond_cond_layer_0_biasadd"layer_0_igdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_0_igdn_0_cond_1_cond_cond_false_203735*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_0_igdn_0_cond_1_cond_cond_true_2037342!
layer_0/igdn_0/cond_1/cond/cond?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity(layer_0/igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity"S
#layer_0_igdn_0_cond_1_cond_identity,layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+synthesis_layer_2_igdn_2_cond_1_true_202989F
Bsynthesis_layer_2_igdn_2_cond_1_identity_synthesis_layer_2_biasadd/
+synthesis_layer_2_igdn_2_cond_1_placeholder,
(synthesis_layer_2_igdn_2_cond_1_identity?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentityBsynthesis_layer_2_igdn_2_cond_1_identity_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity"]
(synthesis_layer_2_igdn_2_cond_1_identity1synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_0_igdn_0_cond_false_2037055
1layer_0_igdn_0_cond_identity_layer_0_igdn_0_equal
 
layer_0_igdn_0_cond_identity
?
layer_0/igdn_0/cond/IdentityIdentity1layer_0_igdn_0_cond_identity_layer_0_igdn_0_equal*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity"E
layer_0_igdn_0_cond_identity%layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
0synthesis_layer_1_igdn_1_cond_1_cond_true_202313F
Bsynthesis_layer_1_igdn_1_cond_1_cond_abs_synthesis_layer_1_biasadd4
0synthesis_layer_1_igdn_1_cond_1_cond_placeholder1
-synthesis_layer_1_igdn_1_cond_1_cond_identity?
(synthesis/layer_1/igdn_1/cond_1/cond/AbsAbsBsynthesis_layer_1_igdn_1_cond_1_cond_abs_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/cond/Abs?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity,synthesis/layer_1/igdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_1_cond_identity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_2_cond_true_204335*
&igdn_0_cond_2_cond_sqrt_igdn_0_biasadd"
igdn_0_cond_2_cond_placeholder
igdn_0_cond_2_cond_identity?
igdn_0/cond_2/cond/SqrtSqrt&igdn_0_cond_2_cond_sqrt_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Sqrt?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity"C
igdn_0_cond_2_cond_identity$igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_0_cond_1_false_204244
igdn_0_cond_1_cond_biasadd
igdn_0_cond_1_equal_x
igdn_0_cond_1_identityg
igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_0/cond_1/x?
igdn_0/cond_1/EqualEqualigdn_0_cond_1_equal_xigdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_1/Equal?
igdn_0/cond_1/condStatelessIfigdn_0/cond_1/Equal:z:0igdn_0_cond_1_cond_biasaddigdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_0_cond_1_cond_false_204253*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_1_cond_true_2042522
igdn_0/cond_1/cond?
igdn_0/cond_1/cond/IdentityIdentityigdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity?
igdn_0/cond_1/IdentityIdentity$igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity"9
igdn_0_cond_1_identityigdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_0_igdn_0_cond_1_false_202668B
>synthesis_layer_0_igdn_0_cond_1_cond_synthesis_layer_0_biasadd+
'synthesis_layer_0_igdn_0_cond_1_equal_x,
(synthesis_layer_0_igdn_0_cond_1_identity?
!synthesis/layer_0/igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_0/igdn_0/cond_1/x?
%synthesis/layer_0/igdn_0/cond_1/EqualEqual'synthesis_layer_0_igdn_0_cond_1_equal_x*synthesis/layer_0/igdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_0/igdn_0/cond_1/Equal?
$synthesis/layer_0/igdn_0/cond_1/condStatelessIf)synthesis/layer_0/igdn_0/cond_1/Equal:z:0>synthesis_layer_0_igdn_0_cond_1_cond_synthesis_layer_0_biasadd'synthesis_layer_0_igdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_0_igdn_0_cond_1_cond_false_202677*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_1_cond_true_2026762&
$synthesis/layer_0/igdn_0/cond_1/cond?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity"]
(synthesis_layer_0_igdn_0_cond_1_identity1synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
??
?
A__inference_model_layer_call_and_return_conditional_losses_203139

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_0_biasadd_readvariableop_resource:	?$
 synthesis_layer_0_igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y&
"synthesis_layer_0_igdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_1_biasadd_readvariableop_resource:	?$
 synthesis_layer_1_igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y&
"synthesis_layer_1_igdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_2_biasadd_readvariableop_resource:	?$
 synthesis_layer_2_igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y&
"synthesis_layer_2_igdn_2_equal_1_x
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	??
1synthesis_layer_3_biasadd_readvariableop_resource:
identity??.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?(synthesis/layer_0/BiasAdd/ReadVariableOp?(synthesis/layer_1/BiasAdd/ReadVariableOp?(synthesis/layer_2/BiasAdd/ReadVariableOp?(synthesis/layer_3/BiasAdd/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshape?
 synthesis/layer_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_0/transpose/perm?
synthesis/layer_0/transpose	Transposelayer_0/kernel/Reshape:output:0)synthesis/layer_0/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_0/transposeh
synthesis/layer_0/ShapeShapeinputs*
T0*
_output_shapes
:2
synthesis/layer_0/Shape?
%synthesis/layer_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_0/strided_slice/stack?
'synthesis/layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice/stack_1?
'synthesis/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice/stack_2?
synthesis/layer_0/strided_sliceStridedSlice synthesis/layer_0/Shape:output:0.synthesis/layer_0/strided_slice/stack:output:00synthesis/layer_0/strided_slice/stack_1:output:00synthesis/layer_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_0/strided_slice?
'synthesis/layer_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice_1/stack?
)synthesis/layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_1/stack_1?
)synthesis/layer_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_1/stack_2?
!synthesis/layer_0/strided_slice_1StridedSlice synthesis/layer_0/Shape:output:00synthesis/layer_0/strided_slice_1/stack:output:02synthesis/layer_0/strided_slice_1/stack_1:output:02synthesis/layer_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_0/strided_slice_1t
synthesis/layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_0/mul/y?
synthesis/layer_0/mulMul*synthesis/layer_0/strided_slice_1:output:0 synthesis/layer_0/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/mult
synthesis/layer_0/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_0/add/y?
synthesis/layer_0/addAddV2synthesis/layer_0/mul:z:0 synthesis/layer_0/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/add?
'synthesis/layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice_2/stack?
)synthesis/layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_2/stack_1?
)synthesis/layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_2/stack_2?
!synthesis/layer_0/strided_slice_2StridedSlice synthesis/layer_0/Shape:output:00synthesis/layer_0/strided_slice_2/stack:output:02synthesis/layer_0/strided_slice_2/stack_1:output:02synthesis/layer_0/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_0/strided_slice_2x
synthesis/layer_0/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_0/mul_1/y?
synthesis/layer_0/mul_1Mul*synthesis/layer_0/strided_slice_2:output:0"synthesis/layer_0/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/mul_1x
synthesis/layer_0/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_0/add_1/y?
synthesis/layer_0/add_1AddV2synthesis/layer_0/mul_1:z:0"synthesis/layer_0/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/add_1?
0synthesis/layer_0/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_0/conv2d_transpose/input_sizes/3?
.synthesis/layer_0/conv2d_transpose/input_sizesPack(synthesis/layer_0/strided_slice:output:0synthesis/layer_0/add:z:0synthesis/layer_0/add_1:z:09synthesis/layer_0/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_0/conv2d_transpose/input_sizes?
"synthesis/layer_0/conv2d_transposeConv2DBackpropInput7synthesis/layer_0/conv2d_transpose/input_sizes:output:0synthesis/layer_0/transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_0/conv2d_transpose?
(synthesis/layer_0/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_0/BiasAdd/ReadVariableOp?
synthesis/layer_0/BiasAddBiasAdd+synthesis/layer_0/conv2d_transpose:output:00synthesis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_0/BiasAdd}
synthesis/layer_0/igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_0/igdn_0/x?
synthesis/layer_0/igdn_0/EqualEqual synthesis_layer_0_igdn_0_equal_x#synthesis/layer_0/igdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_0/igdn_0/Equal?
synthesis/layer_0/igdn_0/condStatelessIf"synthesis/layer_0/igdn_0/Equal:z:0"synthesis/layer_0/igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_0_igdn_0_cond_false_202657*
output_shapes
: *<
then_branch-R+
)synthesis_layer_0_igdn_0_cond_true_2026562
synthesis/layer_0/igdn_0/cond?
&synthesis/layer_0/igdn_0/cond/IdentityIdentity&synthesis/layer_0/igdn_0/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity?
synthesis/layer_0/igdn_0/cond_1StatelessIf/synthesis/layer_0/igdn_0/cond/Identity:output:0"synthesis/layer_0/BiasAdd:output:0 synthesis_layer_0_igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_0_igdn_0_cond_1_false_202668*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_1_true_2026672!
synthesis/layer_0/igdn_0/cond_1?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentity(synthesis/layer_0/igdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202713*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202723*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
&synthesis/layer_0/igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_0/igdn_0/Reshape/shape?
 synthesis/layer_0/igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0/synthesis/layer_0/igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_0/igdn_0/Reshape?
$synthesis/layer_0/igdn_0/convolutionConv2D1synthesis/layer_0/igdn_0/cond_1/Identity:output:0)synthesis/layer_0/igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_0/igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202737*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
 synthesis/layer_0/igdn_0/BiasAddBiasAdd-synthesis/layer_0/igdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_0/igdn_0/BiasAdd?
synthesis/layer_0/igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_0/igdn_0/x_1?
 synthesis/layer_0/igdn_0/Equal_1Equal"synthesis_layer_0_igdn_0_equal_1_x%synthesis/layer_0/igdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_0/igdn_0/Equal_1?
synthesis/layer_0/igdn_0/cond_2StatelessIf$synthesis/layer_0/igdn_0/Equal_1:z:0)synthesis/layer_0/igdn_0/BiasAdd:output:0"synthesis_layer_0_igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_0_igdn_0_cond_2_false_202751*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_2_true_2027502!
synthesis/layer_0/igdn_0/cond_2?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentity(synthesis/layer_0/igdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity?
synthesis/layer_0/igdn_0/mulMul"synthesis/layer_0/BiasAdd:output:01synthesis/layer_0/igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_0/igdn_0/mul?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshape?
 synthesis/layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_1/transpose/perm?
synthesis/layer_1/transpose	Transposelayer_1/kernel/Reshape:output:0)synthesis/layer_1/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_1/transpose?
synthesis/layer_1/ShapeShape synthesis/layer_0/igdn_0/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_1/Shape?
%synthesis/layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_1/strided_slice/stack?
'synthesis/layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice/stack_1?
'synthesis/layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice/stack_2?
synthesis/layer_1/strided_sliceStridedSlice synthesis/layer_1/Shape:output:0.synthesis/layer_1/strided_slice/stack:output:00synthesis/layer_1/strided_slice/stack_1:output:00synthesis/layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_1/strided_slice?
'synthesis/layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice_1/stack?
)synthesis/layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_1/stack_1?
)synthesis/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_1/stack_2?
!synthesis/layer_1/strided_slice_1StridedSlice synthesis/layer_1/Shape:output:00synthesis/layer_1/strided_slice_1/stack:output:02synthesis/layer_1/strided_slice_1/stack_1:output:02synthesis/layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_1/strided_slice_1t
synthesis/layer_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_1/mul/y?
synthesis/layer_1/mulMul*synthesis/layer_1/strided_slice_1:output:0 synthesis/layer_1/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/mult
synthesis/layer_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_1/add/y?
synthesis/layer_1/addAddV2synthesis/layer_1/mul:z:0 synthesis/layer_1/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/add?
'synthesis/layer_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice_2/stack?
)synthesis/layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_2/stack_1?
)synthesis/layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_2/stack_2?
!synthesis/layer_1/strided_slice_2StridedSlice synthesis/layer_1/Shape:output:00synthesis/layer_1/strided_slice_2/stack:output:02synthesis/layer_1/strided_slice_2/stack_1:output:02synthesis/layer_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_1/strided_slice_2x
synthesis/layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_1/mul_1/y?
synthesis/layer_1/mul_1Mul*synthesis/layer_1/strided_slice_2:output:0"synthesis/layer_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/mul_1x
synthesis/layer_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_1/add_1/y?
synthesis/layer_1/add_1AddV2synthesis/layer_1/mul_1:z:0"synthesis/layer_1/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/add_1?
0synthesis/layer_1/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_1/conv2d_transpose/input_sizes/3?
.synthesis/layer_1/conv2d_transpose/input_sizesPack(synthesis/layer_1/strided_slice:output:0synthesis/layer_1/add:z:0synthesis/layer_1/add_1:z:09synthesis/layer_1/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_1/conv2d_transpose/input_sizes?
"synthesis/layer_1/conv2d_transposeConv2DBackpropInput7synthesis/layer_1/conv2d_transpose/input_sizes:output:0synthesis/layer_1/transpose:y:0 synthesis/layer_0/igdn_0/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_1/conv2d_transpose?
(synthesis/layer_1/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_1/BiasAdd/ReadVariableOp?
synthesis/layer_1/BiasAddBiasAdd+synthesis/layer_1/conv2d_transpose:output:00synthesis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_1/BiasAdd}
synthesis/layer_1/igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_1/igdn_1/x?
synthesis/layer_1/igdn_1/EqualEqual synthesis_layer_1_igdn_1_equal_x#synthesis/layer_1/igdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_1/igdn_1/Equal?
synthesis/layer_1/igdn_1/condStatelessIf"synthesis/layer_1/igdn_1/Equal:z:0"synthesis/layer_1/igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_1_igdn_1_cond_false_202818*
output_shapes
: *<
then_branch-R+
)synthesis_layer_1_igdn_1_cond_true_2028172
synthesis/layer_1/igdn_1/cond?
&synthesis/layer_1/igdn_1/cond/IdentityIdentity&synthesis/layer_1/igdn_1/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity?
synthesis/layer_1/igdn_1/cond_1StatelessIf/synthesis/layer_1/igdn_1/cond/Identity:output:0"synthesis/layer_1/BiasAdd:output:0 synthesis_layer_1_igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_1_igdn_1_cond_1_false_202829*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_1_true_2028282!
synthesis/layer_1/igdn_1/cond_1?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentity(synthesis/layer_1/igdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202874*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202884*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
&synthesis/layer_1/igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_1/igdn_1/Reshape/shape?
 synthesis/layer_1/igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0/synthesis/layer_1/igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_1/igdn_1/Reshape?
$synthesis/layer_1/igdn_1/convolutionConv2D1synthesis/layer_1/igdn_1/cond_1/Identity:output:0)synthesis/layer_1/igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_1/igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202898*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
 synthesis/layer_1/igdn_1/BiasAddBiasAdd-synthesis/layer_1/igdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_1/igdn_1/BiasAdd?
synthesis/layer_1/igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_1/igdn_1/x_1?
 synthesis/layer_1/igdn_1/Equal_1Equal"synthesis_layer_1_igdn_1_equal_1_x%synthesis/layer_1/igdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_1/igdn_1/Equal_1?
synthesis/layer_1/igdn_1/cond_2StatelessIf$synthesis/layer_1/igdn_1/Equal_1:z:0)synthesis/layer_1/igdn_1/BiasAdd:output:0"synthesis_layer_1_igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_1_igdn_1_cond_2_false_202912*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_2_true_2029112!
synthesis/layer_1/igdn_1/cond_2?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentity(synthesis/layer_1/igdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity?
synthesis/layer_1/igdn_1/mulMul"synthesis/layer_1/BiasAdd:output:01synthesis/layer_1/igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_1/igdn_1/mul?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshape?
 synthesis/layer_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_2/transpose/perm?
synthesis/layer_2/transpose	Transposelayer_2/kernel/Reshape:output:0)synthesis/layer_2/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_2/transpose?
synthesis/layer_2/ShapeShape synthesis/layer_1/igdn_1/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_2/Shape?
%synthesis/layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_2/strided_slice/stack?
'synthesis/layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice/stack_1?
'synthesis/layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice/stack_2?
synthesis/layer_2/strided_sliceStridedSlice synthesis/layer_2/Shape:output:0.synthesis/layer_2/strided_slice/stack:output:00synthesis/layer_2/strided_slice/stack_1:output:00synthesis/layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_2/strided_slice?
'synthesis/layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice_1/stack?
)synthesis/layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_1/stack_1?
)synthesis/layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_1/stack_2?
!synthesis/layer_2/strided_slice_1StridedSlice synthesis/layer_2/Shape:output:00synthesis/layer_2/strided_slice_1/stack:output:02synthesis/layer_2/strided_slice_1/stack_1:output:02synthesis/layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_2/strided_slice_1t
synthesis/layer_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_2/mul/y?
synthesis/layer_2/mulMul*synthesis/layer_2/strided_slice_1:output:0 synthesis/layer_2/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/mult
synthesis/layer_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_2/add/y?
synthesis/layer_2/addAddV2synthesis/layer_2/mul:z:0 synthesis/layer_2/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/add?
'synthesis/layer_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice_2/stack?
)synthesis/layer_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_2/stack_1?
)synthesis/layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_2/stack_2?
!synthesis/layer_2/strided_slice_2StridedSlice synthesis/layer_2/Shape:output:00synthesis/layer_2/strided_slice_2/stack:output:02synthesis/layer_2/strided_slice_2/stack_1:output:02synthesis/layer_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_2/strided_slice_2x
synthesis/layer_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_2/mul_1/y?
synthesis/layer_2/mul_1Mul*synthesis/layer_2/strided_slice_2:output:0"synthesis/layer_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/mul_1x
synthesis/layer_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_2/add_1/y?
synthesis/layer_2/add_1AddV2synthesis/layer_2/mul_1:z:0"synthesis/layer_2/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/add_1?
0synthesis/layer_2/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_2/conv2d_transpose/input_sizes/3?
.synthesis/layer_2/conv2d_transpose/input_sizesPack(synthesis/layer_2/strided_slice:output:0synthesis/layer_2/add:z:0synthesis/layer_2/add_1:z:09synthesis/layer_2/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_2/conv2d_transpose/input_sizes?
"synthesis/layer_2/conv2d_transposeConv2DBackpropInput7synthesis/layer_2/conv2d_transpose/input_sizes:output:0synthesis/layer_2/transpose:y:0 synthesis/layer_1/igdn_1/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_2/conv2d_transpose?
(synthesis/layer_2/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_2/BiasAdd/ReadVariableOp?
synthesis/layer_2/BiasAddBiasAdd+synthesis/layer_2/conv2d_transpose:output:00synthesis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_2/BiasAdd}
synthesis/layer_2/igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_2/igdn_2/x?
synthesis/layer_2/igdn_2/EqualEqual synthesis_layer_2_igdn_2_equal_x#synthesis/layer_2/igdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_2/igdn_2/Equal?
synthesis/layer_2/igdn_2/condStatelessIf"synthesis/layer_2/igdn_2/Equal:z:0"synthesis/layer_2/igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_2_igdn_2_cond_false_202979*
output_shapes
: *<
then_branch-R+
)synthesis_layer_2_igdn_2_cond_true_2029782
synthesis/layer_2/igdn_2/cond?
&synthesis/layer_2/igdn_2/cond/IdentityIdentity&synthesis/layer_2/igdn_2/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity?
synthesis/layer_2/igdn_2/cond_1StatelessIf/synthesis/layer_2/igdn_2/cond/Identity:output:0"synthesis/layer_2/BiasAdd:output:0 synthesis_layer_2_igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_2_igdn_2_cond_1_false_202990*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_1_true_2029892!
synthesis/layer_2/igdn_2/cond_1?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentity(synthesis/layer_2/igdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203035*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203045*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
&synthesis/layer_2/igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_2/igdn_2/Reshape/shape?
 synthesis/layer_2/igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0/synthesis/layer_2/igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_2/igdn_2/Reshape?
$synthesis/layer_2/igdn_2/convolutionConv2D1synthesis/layer_2/igdn_2/cond_1/Identity:output:0)synthesis/layer_2/igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_2/igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203059*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
 synthesis/layer_2/igdn_2/BiasAddBiasAdd-synthesis/layer_2/igdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_2/igdn_2/BiasAdd?
synthesis/layer_2/igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_2/igdn_2/x_1?
 synthesis/layer_2/igdn_2/Equal_1Equal"synthesis_layer_2_igdn_2_equal_1_x%synthesis/layer_2/igdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_2/igdn_2/Equal_1?
synthesis/layer_2/igdn_2/cond_2StatelessIf$synthesis/layer_2/igdn_2/Equal_1:z:0)synthesis/layer_2/igdn_2/BiasAdd:output:0"synthesis_layer_2_igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_2_igdn_2_cond_2_false_203073*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_2_true_2030722!
synthesis/layer_2/igdn_2/cond_2?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentity(synthesis/layer_2/igdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity?
synthesis/layer_2/igdn_2/mulMul"synthesis/layer_2/BiasAdd:output:01synthesis/layer_2/igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_2/igdn_2/mul?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshape?
 synthesis/layer_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_3/transpose/perm?
synthesis/layer_3/transpose	Transposelayer_3/kernel/Reshape:output:0)synthesis/layer_3/transpose/perm:output:0*
T0*'
_output_shapes
:?2
synthesis/layer_3/transpose?
synthesis/layer_3/ShapeShape synthesis/layer_2/igdn_2/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_3/Shape?
%synthesis/layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_3/strided_slice/stack?
'synthesis/layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice/stack_1?
'synthesis/layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice/stack_2?
synthesis/layer_3/strided_sliceStridedSlice synthesis/layer_3/Shape:output:0.synthesis/layer_3/strided_slice/stack:output:00synthesis/layer_3/strided_slice/stack_1:output:00synthesis/layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_3/strided_slice?
'synthesis/layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice_1/stack?
)synthesis/layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_1/stack_1?
)synthesis/layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_1/stack_2?
!synthesis/layer_3/strided_slice_1StridedSlice synthesis/layer_3/Shape:output:00synthesis/layer_3/strided_slice_1/stack:output:02synthesis/layer_3/strided_slice_1/stack_1:output:02synthesis/layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_3/strided_slice_1t
synthesis/layer_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_3/mul/y?
synthesis/layer_3/mulMul*synthesis/layer_3/strided_slice_1:output:0 synthesis/layer_3/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/mult
synthesis/layer_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_3/add/y?
synthesis/layer_3/addAddV2synthesis/layer_3/mul:z:0 synthesis/layer_3/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/add?
'synthesis/layer_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice_2/stack?
)synthesis/layer_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_2/stack_1?
)synthesis/layer_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_2/stack_2?
!synthesis/layer_3/strided_slice_2StridedSlice synthesis/layer_3/Shape:output:00synthesis/layer_3/strided_slice_2/stack:output:02synthesis/layer_3/strided_slice_2/stack_1:output:02synthesis/layer_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_3/strided_slice_2x
synthesis/layer_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_3/mul_1/y?
synthesis/layer_3/mul_1Mul*synthesis/layer_3/strided_slice_2:output:0"synthesis/layer_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/mul_1x
synthesis/layer_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_3/add_1/y?
synthesis/layer_3/add_1AddV2synthesis/layer_3/mul_1:z:0"synthesis/layer_3/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/add_1?
0synthesis/layer_3/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :22
0synthesis/layer_3/conv2d_transpose/input_sizes/3?
.synthesis/layer_3/conv2d_transpose/input_sizesPack(synthesis/layer_3/strided_slice:output:0synthesis/layer_3/add:z:0synthesis/layer_3/add_1:z:09synthesis/layer_3/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_3/conv2d_transpose/input_sizes?
"synthesis/layer_3/conv2d_transposeConv2DBackpropInput7synthesis/layer_3/conv2d_transpose/input_sizes:output:0synthesis/layer_3/transpose:y:0 synthesis/layer_2/igdn_2/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_3/conv2d_transpose?
(synthesis/layer_3/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(synthesis/layer_3/BiasAdd/ReadVariableOp?
synthesis/layer_3/BiasAddBiasAdd+synthesis/layer_3/conv2d_transpose:output:00synthesis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2
synthesis/layer_3/BiasAddy
synthesis/lambda_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
synthesis/lambda_1/mul/y?
synthesis/lambda_1/mulMul"synthesis/layer_3/BiasAdd:output:0!synthesis/lambda_1/mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
synthesis/lambda_1/mul?
IdentityIdentitysynthesis/lambda_1/mul:z:0/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp)^synthesis/layer_0/BiasAdd/ReadVariableOp)^synthesis/layer_1/BiasAdd/ReadVariableOp)^synthesis/layer_2/BiasAdd/ReadVariableOp)^synthesis/layer_3/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp2T
(synthesis/layer_0/BiasAdd/ReadVariableOp(synthesis/layer_0/BiasAdd/ReadVariableOp2T
(synthesis/layer_1/BiasAdd/ReadVariableOp(synthesis/layer_1/BiasAdd/ReadVariableOp2T
(synthesis/layer_2/BiasAdd/ReadVariableOp(synthesis/layer_2/BiasAdd/ReadVariableOp2T
(synthesis/layer_3/BiasAdd/ReadVariableOp(synthesis/layer_3/BiasAdd/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
h
layer_1_igdn_1_cond_true_203341#
layer_1_igdn_1_cond_placeholder
 
layer_1_igdn_1_cond_identity
x
layer_1/igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_1/igdn_1/cond/Const?
layer_1/igdn_1/cond/IdentityIdentity"layer_1/igdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity"E
layer_1_igdn_1_cond_identity%layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_1_igdn_1_cond_2_true_202911M
Isynthesis_layer_1_igdn_1_cond_2_identity_synthesis_layer_1_igdn_1_biasadd/
+synthesis_layer_1_igdn_1_cond_2_placeholder,
(synthesis_layer_1_igdn_1_cond_2_identity?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentityIsynthesis_layer_1_igdn_1_cond_2_identity_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity"]
(synthesis_layer_1_igdn_1_cond_2_identity1synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_2_cond_2_cond_false_204674)
%igdn_2_cond_2_cond_pow_igdn_2_biasadd
igdn_2_cond_2_cond_pow_y
igdn_2_cond_2_cond_identity?
igdn_2/cond_2/cond/powPow%igdn_2_cond_2_cond_pow_igdn_2_biasaddigdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/pow?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity"C
igdn_2_cond_2_cond_identity$igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_1_igdn_1_cond_1_cond_false_202838G
Csynthesis_layer_1_igdn_1_cond_1_cond_cond_synthesis_layer_1_biasadd0
,synthesis_layer_1_igdn_1_cond_1_cond_equal_x1
-synthesis_layer_1_igdn_1_cond_1_cond_identity?
&synthesis/layer_1/igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_1/igdn_1/cond_1/cond/x?
*synthesis/layer_1/igdn_1/cond_1/cond/EqualEqual,synthesis_layer_1_igdn_1_cond_1_cond_equal_x/synthesis/layer_1/igdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_1/igdn_1/cond_1/cond/Equal?
)synthesis/layer_1/igdn_1/cond_1/cond/condStatelessIf.synthesis/layer_1/igdn_1/cond_1/cond/Equal:z:0Csynthesis_layer_1_igdn_1_cond_1_cond_cond_synthesis_layer_1_biasadd,synthesis_layer_1_igdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202848*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_2028472+
)synthesis/layer_1/igdn_1/cond_1/cond/cond?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity2synthesis/layer_1/igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_1_cond_identity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1synthesis_layer_2_igdn_2_cond_1_cond_false_202999G
Csynthesis_layer_2_igdn_2_cond_1_cond_cond_synthesis_layer_2_biasadd0
,synthesis_layer_2_igdn_2_cond_1_cond_equal_x1
-synthesis_layer_2_igdn_2_cond_1_cond_identity?
&synthesis/layer_2/igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_2/igdn_2/cond_1/cond/x?
*synthesis/layer_2/igdn_2/cond_1/cond/EqualEqual,synthesis_layer_2_igdn_2_cond_1_cond_equal_x/synthesis/layer_2/igdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_2/igdn_2/cond_1/cond/Equal?
)synthesis/layer_2/igdn_2/cond_1/cond/condStatelessIf.synthesis/layer_2/igdn_2/cond_1/cond/Equal:z:0Csynthesis_layer_2_igdn_2_cond_1_cond_cond_synthesis_layer_2_biasadd,synthesis_layer_2_igdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_203009*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_2030082+
)synthesis/layer_2/igdn_2/cond_1/cond/cond?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity2synthesis/layer_2/igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_1_cond_identity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
2model_synthesis_layer_0_igdn_0_cond_1_false_200046N
Jmodel_synthesis_layer_0_igdn_0_cond_1_cond_model_synthesis_layer_0_biasadd1
-model_synthesis_layer_0_igdn_0_cond_1_equal_x2
.model_synthesis_layer_0_igdn_0_cond_1_identity?
'model/synthesis/layer_0/igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'model/synthesis/layer_0/igdn_0/cond_1/x?
+model/synthesis/layer_0/igdn_0/cond_1/EqualEqual-model_synthesis_layer_0_igdn_0_cond_1_equal_x0model/synthesis/layer_0/igdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_0/igdn_0/cond_1/Equal?
*model/synthesis/layer_0/igdn_0/cond_1/condStatelessIf/model/synthesis/layer_0/igdn_0/cond_1/Equal:z:0Jmodel_synthesis_layer_0_igdn_0_cond_1_cond_model_synthesis_layer_0_biasadd-model_synthesis_layer_0_igdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_0_igdn_0_cond_1_cond_false_200055*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_0_igdn_0_cond_1_cond_true_2000542,
*model/synthesis/layer_0/igdn_0/cond_1/cond?
3model/synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity3model/synthesis/layer_0/igdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_1/cond/Identity?
.model/synthesis/layer_0/igdn_0/cond_1/IdentityIdentity<model/synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_1/Identity"i
.model_synthesis_layer_0_igdn_0_cond_1_identity7model/synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_2_igdn_2_cond_1_false_202990B
>synthesis_layer_2_igdn_2_cond_1_cond_synthesis_layer_2_biasadd+
'synthesis_layer_2_igdn_2_cond_1_equal_x,
(synthesis_layer_2_igdn_2_cond_1_identity?
!synthesis/layer_2/igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_2/igdn_2/cond_1/x?
%synthesis/layer_2/igdn_2/cond_1/EqualEqual'synthesis_layer_2_igdn_2_cond_1_equal_x*synthesis/layer_2/igdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_2/igdn_2/cond_1/Equal?
$synthesis/layer_2/igdn_2/cond_1/condStatelessIf)synthesis/layer_2/igdn_2/cond_1/Equal:z:0>synthesis_layer_2_igdn_2_cond_1_cond_synthesis_layer_2_biasadd'synthesis_layer_2_igdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_2_igdn_2_cond_1_cond_false_202999*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_1_cond_true_2029982&
$synthesis/layer_2/igdn_2/cond_1/cond?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity"]
(synthesis_layer_2_igdn_2_cond_1_identity1synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_1_cond_1_true_204412"
igdn_1_cond_1_identity_biasadd
igdn_1_cond_1_placeholder
igdn_1_cond_1_identity?
igdn_1/cond_1/IdentityIdentityigdn_1_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity"9
igdn_1_cond_1_identityigdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1model_synthesis_layer_1_igdn_1_cond_1_true_200206R
Nmodel_synthesis_layer_1_igdn_1_cond_1_identity_model_synthesis_layer_1_biasadd5
1model_synthesis_layer_1_igdn_1_cond_1_placeholder2
.model_synthesis_layer_1_igdn_1_cond_1_identity?
.model/synthesis/layer_1/igdn_1/cond_1/IdentityIdentityNmodel_synthesis_layer_1_igdn_1_cond_1_identity_model_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_1/Identity"i
.model_synthesis_layer_1_igdn_1_cond_1_identity7model/synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
h
layer_2_igdn_2_cond_true_204026#
layer_2_igdn_2_cond_placeholder
 
layer_2_igdn_2_cond_identity
x
layer_2/igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_2/igdn_2/cond/Const?
layer_2/igdn_2/cond/IdentityIdentity"layer_2/igdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity"E
layer_2_igdn_2_cond_identity%layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
!layer_2_igdn_2_cond_2_true_2041209
5layer_2_igdn_2_cond_2_identity_layer_2_igdn_2_biasadd%
!layer_2_igdn_2_cond_2_placeholder"
layer_2_igdn_2_cond_2_identity?
layer_2/igdn_2/cond_2/IdentityIdentity5layer_2_igdn_2_cond_2_identity_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity"I
layer_2_igdn_2_cond_2_identity'layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_1_cond_1_false_200763
igdn_1_cond_1_cond_biasadd
igdn_1_cond_1_equal_x
igdn_1_cond_1_identityg
igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_1/cond_1/x?
igdn_1/cond_1/EqualEqualigdn_1_cond_1_equal_xigdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_1/Equal?
igdn_1/cond_1/condStatelessIfigdn_1/cond_1/Equal:z:0igdn_1_cond_1_cond_biasaddigdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_1_cond_1_cond_false_200772*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_1_cond_true_2007712
igdn_1/cond_1/cond?
igdn_1/cond_1/cond/IdentityIdentityigdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity?
igdn_1/cond_1/IdentityIdentity$igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity"9
igdn_1_cond_1_identityigdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_0_cond_1_cond_cond_true_204262*
&igdn_0_cond_1_cond_cond_square_biasadd'
#igdn_0_cond_1_cond_cond_placeholder$
 igdn_0_cond_1_cond_cond_identity?
igdn_0/cond_1/cond/cond/SquareSquare&igdn_0_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_0/cond_1/cond/cond/Square?
 igdn_0/cond_1/cond/cond/IdentityIdentity"igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity"M
 igdn_0_cond_1_cond_cond_identity)igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?i
?
C__inference_layer_2_layer_call_and_return_conditional_losses_201064

inputs
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y
igdn_2_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_2/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_2/x?
igdn_2/EqualEqualigdn_2_equal_xigdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/Equal?
igdn_2/condStatelessIfigdn_2/Equal:z:0igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_2_cond_false_200941*
output_shapes
: **
then_branchR
igdn_2_cond_true_2009402
igdn_2/condo
igdn_2/cond/IdentityIdentityigdn_2/cond:output:0*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity?
igdn_2/cond_1StatelessIfigdn_2/cond/Identity:output:0BiasAdd:output:0igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_2_cond_1_false_200952*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_1_true_2009512
igdn_2/cond_1?
igdn_2/cond_1/IdentityIdentityigdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200997*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-201007*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_2/Reshape/shape?
igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_2/Reshape?
igdn_2/convolutionConv2Digdn_2/cond_1/Identity:output:0igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-201021*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
igdn_2/BiasAddBiasAddigdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/BiasAdd]

igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_2/x_1?
igdn_2/Equal_1Equaligdn_2_equal_1_xigdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/Equal_1?
igdn_2/cond_2StatelessIfigdn_2/Equal_1:z:0igdn_2/BiasAdd:output:0igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_2_cond_2_false_201035*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_2_true_2010342
igdn_2/cond_2?
igdn_2/cond_2/IdentityIdentityigdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity?

igdn_2/mulMulBiasAdd:output:0igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_2/mul?
IdentityIdentityigdn_2/mul:z:0^BiasAdd/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
+synthesis_layer_2_igdn_2_cond_1_true_202465F
Bsynthesis_layer_2_igdn_2_cond_1_identity_synthesis_layer_2_biasadd/
+synthesis_layer_2_igdn_2_cond_1_placeholder,
(synthesis_layer_2_igdn_2_cond_1_identity?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentityBsynthesis_layer_2_igdn_2_cond_1_identity_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity"]
(synthesis_layer_2_igdn_2_cond_1_identity1synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$igdn_2_cond_1_cond_cond_false_204601'
#igdn_2_cond_1_cond_cond_pow_biasadd!
igdn_2_cond_1_cond_cond_pow_y$
 igdn_2_cond_1_cond_cond_identity?
igdn_2/cond_1/cond/cond/powPow#igdn_2_cond_1_cond_cond_pow_biasaddigdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/cond/pow?
 igdn_2/cond_1/cond/cond/IdentityIdentityigdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity"M
 igdn_2_cond_1_cond_cond_identity)igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
/model_synthesis_layer_2_igdn_2_cond_true_2003563
/model_synthesis_layer_2_igdn_2_cond_placeholder
0
,model_synthesis_layer_2_igdn_2_cond_identity
?
)model/synthesis/layer_2/igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/synthesis/layer_2/igdn_2/cond/Const?
,model/synthesis/layer_2/igdn_2/cond/IdentityIdentity2model/synthesis/layer_2/igdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_2/igdn_2/cond/Identity"e
,model_synthesis_layer_2_igdn_2_cond_identity5model/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
"layer_2_igdn_2_cond_2_false_2041215
1layer_2_igdn_2_cond_2_cond_layer_2_igdn_2_biasadd!
layer_2_igdn_2_cond_2_equal_x"
layer_2_igdn_2_cond_2_identityw
layer_2/igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_2/igdn_2/cond_2/x?
layer_2/igdn_2/cond_2/EqualEquallayer_2_igdn_2_cond_2_equal_x layer_2/igdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/cond_2/Equal?
layer_2/igdn_2/cond_2/condStatelessIflayer_2/igdn_2/cond_2/Equal:z:01layer_2_igdn_2_cond_2_cond_layer_2_igdn_2_biasaddlayer_2_igdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_2_igdn_2_cond_2_cond_false_204130*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_2_cond_true_2041292
layer_2/igdn_2/cond_2/cond?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity#layer_2/igdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity?
layer_2/igdn_2/cond_2/IdentityIdentity,layer_2/igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity"I
layer_2_igdn_2_cond_2_identity'layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$igdn_1_cond_1_cond_cond_false_200782'
#igdn_1_cond_1_cond_cond_pow_biasadd!
igdn_1_cond_1_cond_cond_pow_y$
 igdn_1_cond_1_cond_cond_identity?
igdn_1/cond_1/cond/cond/powPow#igdn_1_cond_1_cond_cond_pow_biasaddigdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/cond/pow?
 igdn_1/cond_1/cond/cond/IdentityIdentityigdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity"M
 igdn_1_cond_1_cond_cond_identity)igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_201141

inputs
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
mul/yu
mulMulinputsmul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
*synthesis_layer_0_igdn_0_cond_false_202657I
Esynthesis_layer_0_igdn_0_cond_identity_synthesis_layer_0_igdn_0_equal
*
&synthesis_layer_0_igdn_0_cond_identity
?
&synthesis/layer_0/igdn_0/cond/IdentityIdentityEsynthesis_layer_0_igdn_0_cond_identity_synthesis_layer_0_igdn_0_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity"Y
&synthesis_layer_0_igdn_0_cond_identity/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?i
?
C__inference_layer_1_layer_call_and_return_conditional_losses_204525

inputs
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y
igdn_1_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_1/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_1/x?
igdn_1/EqualEqualigdn_1_equal_xigdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/Equal?
igdn_1/condStatelessIfigdn_1/Equal:z:0igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_1_cond_false_204402*
output_shapes
: **
then_branchR
igdn_1_cond_true_2044012
igdn_1/condo
igdn_1/cond/IdentityIdentityigdn_1/cond:output:0*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity?
igdn_1/cond_1StatelessIfigdn_1/cond/Identity:output:0BiasAdd:output:0igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_1_cond_1_false_204413*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_1_true_2044122
igdn_1/cond_1?
igdn_1/cond_1/IdentityIdentityigdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204458*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204468*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_1/Reshape/shape?
igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_1/Reshape?
igdn_1/convolutionConv2Digdn_1/cond_1/Identity:output:0igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-204482*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
igdn_1/BiasAddBiasAddigdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/BiasAdd]

igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_1/x_1?
igdn_1/Equal_1Equaligdn_1_equal_1_xigdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/Equal_1?
igdn_1/cond_2StatelessIfigdn_1/Equal_1:z:0igdn_1/BiasAdd:output:0igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_1_cond_2_false_204496*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_2_true_2044952
igdn_1/cond_2?
igdn_1/cond_2/IdentityIdentityigdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity?

igdn_1/mulMulBiasAdd:output:0igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_1/mul?
IdentityIdentityigdn_1/mul:z:0^BiasAdd/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
"layer_1_igdn_1_cond_1_false_203353.
*layer_1_igdn_1_cond_1_cond_layer_1_biasadd!
layer_1_igdn_1_cond_1_equal_x"
layer_1_igdn_1_cond_1_identityw
layer_1/igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/cond_1/x?
layer_1/igdn_1/cond_1/EqualEquallayer_1_igdn_1_cond_1_equal_x layer_1/igdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/cond_1/Equal?
layer_1/igdn_1/cond_1/condStatelessIflayer_1/igdn_1/cond_1/Equal:z:0*layer_1_igdn_1_cond_1_cond_layer_1_biasaddlayer_1_igdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_1_igdn_1_cond_1_cond_false_203362*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_1_cond_true_2033612
layer_1/igdn_1/cond_1/cond?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity#layer_1/igdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity?
layer_1/igdn_1/cond_1/IdentityIdentity,layer_1/igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity"I
layer_1_igdn_1_cond_1_identity'layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
)synthesis_layer_2_igdn_2_cond_true_202978-
)synthesis_layer_2_igdn_2_cond_placeholder
*
&synthesis_layer_2_igdn_2_cond_identity
?
#synthesis/layer_2/igdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_2/igdn_2/cond/Const?
&synthesis/layer_2/igdn_2/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity"Y
&synthesis_layer_2_igdn_2_cond_identity/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
,synthesis_layer_2_igdn_2_cond_2_false_202549I
Esynthesis_layer_2_igdn_2_cond_2_cond_synthesis_layer_2_igdn_2_biasadd+
'synthesis_layer_2_igdn_2_cond_2_equal_x,
(synthesis_layer_2_igdn_2_cond_2_identity?
!synthesis/layer_2/igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_2/igdn_2/cond_2/x?
%synthesis/layer_2/igdn_2/cond_2/EqualEqual'synthesis_layer_2_igdn_2_cond_2_equal_x*synthesis/layer_2/igdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_2/igdn_2/cond_2/Equal?
$synthesis/layer_2/igdn_2/cond_2/condStatelessIf)synthesis/layer_2/igdn_2/cond_2/Equal:z:0Esynthesis_layer_2_igdn_2_cond_2_cond_synthesis_layer_2_igdn_2_biasadd'synthesis_layer_2_igdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_2_igdn_2_cond_2_cond_false_202558*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_2_cond_true_2025572&
$synthesis/layer_2/igdn_2/cond_2/cond?
-synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity-synthesis/layer_2/igdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_2/cond/Identity?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentity6synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity"]
(synthesis_layer_2_igdn_2_cond_2_identity1synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6model_synthesis_layer_1_igdn_1_cond_2_cond_true_200298Z
Vmodel_synthesis_layer_1_igdn_1_cond_2_cond_sqrt_model_synthesis_layer_1_igdn_1_biasadd:
6model_synthesis_layer_1_igdn_1_cond_2_cond_placeholder7
3model_synthesis_layer_1_igdn_1_cond_2_cond_identity?
/model/synthesis/layer_1/igdn_1/cond_2/cond/SqrtSqrtVmodel_synthesis_layer_1_igdn_1_cond_2_cond_sqrt_model_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/model/synthesis/layer_1/igdn_1/cond_2/cond/Sqrt?
3model/synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity3model/synthesis/layer_1/igdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_2/cond/Identity"s
3model_synthesis_layer_1_igdn_1_cond_2_cond_identity<model/synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202485K
Gsynthesis_layer_2_igdn_2_cond_1_cond_cond_pow_synthesis_layer_2_biasadd3
/synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y6
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
-synthesis/layer_2/igdn_2/cond_1/cond/cond/powPowGsynthesis_layer_2_igdn_2_cond_1_cond_cond_pow_synthesis_layer_2_biasadd/synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/cond/pow?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity1synthesis/layer_2/igdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"q
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_2_igdn_2_cond_1_cond_true_2040462
.layer_2_igdn_2_cond_1_cond_abs_layer_2_biasadd*
&layer_2_igdn_2_cond_1_cond_placeholder'
#layer_2_igdn_2_cond_1_cond_identity?
layer_2/igdn_2/cond_1/cond/AbsAbs.layer_2_igdn_2_cond_1_cond_abs_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/cond/Abs?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity"layer_2/igdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity"S
#layer_2_igdn_2_cond_1_cond_identity,layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_203008N
Jsynthesis_layer_2_igdn_2_cond_1_cond_cond_square_synthesis_layer_2_biasadd9
5synthesis_layer_2_igdn_2_cond_1_cond_cond_placeholder6
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
0synthesis/layer_2/igdn_2/cond_1/cond/cond/SquareSquareJsynthesis_layer_2_igdn_2_cond_1_cond_cond_square_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_2/igdn_2/cond_1/cond/cond/Square?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity4synthesis/layer_2/igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"q
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_2_igdn_2_cond_false_2035035
1layer_2_igdn_2_cond_identity_layer_2_igdn_2_equal
 
layer_2_igdn_2_cond_identity
?
layer_2/igdn_2/cond/IdentityIdentity1layer_2_igdn_2_cond_identity_layer_2_igdn_2_equal*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity"E
layer_2_igdn_2_cond_identity%layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
&__inference_model_layer_call_fn_201858
input_2
unknown
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3:
??
	unknown_4
	unknown_5
	unknown_6:	?
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:
??

unknown_12:	?

unknown_13

unknown_14:
??

unknown_15

unknown_16

unknown_17:	?

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22:
??

unknown_23:	?

unknown_24

unknown_25:
??

unknown_26

unknown_27

unknown_28:	?

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33:	?

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2017832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
A__inference_model_layer_call_and_return_conditional_losses_201937

inputs
synthesis_201863$
synthesis_201865:
??
synthesis_201867:	?
synthesis_201869$
synthesis_201871:
??
synthesis_201873
synthesis_201875
synthesis_201877:	?
synthesis_201879
synthesis_201881
synthesis_201883
synthesis_201885$
synthesis_201887:
??
synthesis_201889:	?
synthesis_201891$
synthesis_201893:
??
synthesis_201895
synthesis_201897
synthesis_201899:	?
synthesis_201901
synthesis_201903
synthesis_201905
synthesis_201907$
synthesis_201909:
??
synthesis_201911:	?
synthesis_201913$
synthesis_201915:
??
synthesis_201917
synthesis_201919
synthesis_201921:	?
synthesis_201923
synthesis_201925
synthesis_201927
synthesis_201929#
synthesis_201931:	?
synthesis_201933:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinputssynthesis_201863synthesis_201865synthesis_201867synthesis_201869synthesis_201871synthesis_201873synthesis_201875synthesis_201877synthesis_201879synthesis_201881synthesis_201883synthesis_201885synthesis_201887synthesis_201889synthesis_201891synthesis_201893synthesis_201895synthesis_201897synthesis_201899synthesis_201901synthesis_201903synthesis_201905synthesis_201907synthesis_201909synthesis_201911synthesis_201913synthesis_201915synthesis_201917synthesis_201919synthesis_201921synthesis_201923synthesis_201925synthesis_201927synthesis_201929synthesis_201931synthesis_201933*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2014732#
!synthesis/StatefulPartitionedCall?
IdentityIdentity*synthesis/StatefulPartitionedCall:output:0"^synthesis/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2F
!synthesis/StatefulPartitionedCall!synthesis/StatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
6model_synthesis_layer_1_igdn_1_cond_1_cond_true_200215R
Nmodel_synthesis_layer_1_igdn_1_cond_1_cond_abs_model_synthesis_layer_1_biasadd:
6model_synthesis_layer_1_igdn_1_cond_1_cond_placeholder7
3model_synthesis_layer_1_igdn_1_cond_1_cond_identity?
.model/synthesis/layer_1/igdn_1/cond_1/cond/AbsAbsNmodel_synthesis_layer_1_igdn_1_cond_1_cond_abs_model_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_1/cond/Abs?
3model/synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity2model/synthesis/layer_1/igdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_1/cond/Identity"s
3model_synthesis_layer_1_igdn_1_cond_1_cond_identity<model/synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_0_cond_2_false_204327%
!igdn_0_cond_2_cond_igdn_0_biasadd
igdn_0_cond_2_equal_x
igdn_0_cond_2_identityg
igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_0/cond_2/x?
igdn_0/cond_2/EqualEqualigdn_0_cond_2_equal_xigdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_2/Equal?
igdn_0/cond_2/condStatelessIfigdn_0/cond_2/Equal:z:0!igdn_0_cond_2_cond_igdn_0_biasaddigdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_0_cond_2_cond_false_204336*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_2_cond_true_2043352
igdn_0/cond_2/cond?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity?
igdn_0/cond_2/IdentityIdentity$igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity"9
igdn_0_cond_2_identityigdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_1_cond_true_204252"
igdn_0_cond_1_cond_abs_biasadd"
igdn_0_cond_1_cond_placeholder
igdn_0_cond_1_cond_identity?
igdn_0/cond_1/cond/AbsAbsigdn_0_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Abs?
igdn_0/cond_1/cond/IdentityIdentityigdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity"C
igdn_0_cond_1_cond_identity$igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_204743

inputs
identityS
mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
mul/yu
mulMulinputsmul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
layer_1_igdn_1_cond_true_203865#
layer_1_igdn_1_cond_placeholder
 
layer_1_igdn_1_cond_identity
x
layer_1/igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_1/igdn_1/cond/Const?
layer_1/igdn_1/cond/IdentityIdentity"layer_1/igdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity"E
layer_1_igdn_1_cond_identity%layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?	
?
;model_synthesis_layer_0_igdn_0_cond_1_cond_cond_true_200064Z
Vmodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_square_model_synthesis_layer_0_biasadd?
;model_synthesis_layer_0_igdn_0_cond_1_cond_cond_placeholder<
8model_synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
6model/synthesis/layer_0/igdn_0/cond_1/cond/cond/SquareSquareVmodel_synthesis_layer_0_igdn_0_cond_1_cond_cond_square_model_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6model/synthesis/layer_0/igdn_0/cond_1/cond/cond/Square?
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity:model/synthesis/layer_0/igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"}
8model_synthesis_layer_0_igdn_0_cond_1_cond_cond_identityAmodel/synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_1_igdn_1_cond_1_cond_false_2038863
/layer_1_igdn_1_cond_1_cond_cond_layer_1_biasadd&
"layer_1_igdn_1_cond_1_cond_equal_x'
#layer_1_igdn_1_cond_1_cond_identity?
layer_1/igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_1/igdn_1/cond_1/cond/x?
 layer_1/igdn_1/cond_1/cond/EqualEqual"layer_1_igdn_1_cond_1_cond_equal_x%layer_1/igdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_1/igdn_1/cond_1/cond/Equal?
layer_1/igdn_1/cond_1/cond/condStatelessIf$layer_1/igdn_1/cond_1/cond/Equal:z:0/layer_1_igdn_1_cond_1_cond_cond_layer_1_biasadd"layer_1_igdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_1_igdn_1_cond_1_cond_cond_false_203896*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_1_igdn_1_cond_1_cond_cond_true_2038952!
layer_1/igdn_1/cond_1/cond/cond?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity(layer_1/igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity"S
#layer_1_igdn_1_cond_1_cond_identity,layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_2_igdn_2_cond_2_true_2035969
5layer_2_igdn_2_cond_2_identity_layer_2_igdn_2_biasadd%
!layer_2_igdn_2_cond_2_placeholder"
layer_2_igdn_2_cond_2_identity?
layer_2/igdn_2/cond_2/IdentityIdentity5layer_2_igdn_2_cond_2_identity_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity"I
layer_2_igdn_2_cond_2_identity'layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_2_igdn_2_cond_2_cond_false_2036069
5layer_2_igdn_2_cond_2_cond_pow_layer_2_igdn_2_biasadd$
 layer_2_igdn_2_cond_2_cond_pow_y'
#layer_2_igdn_2_cond_2_cond_identity?
layer_2/igdn_2/cond_2/cond/powPow5layer_2_igdn_2_cond_2_cond_pow_layer_2_igdn_2_biasadd layer_2_igdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/cond/pow?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity"layer_2/igdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity"S
#layer_2_igdn_2_cond_2_cond_identity,layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+synthesis_layer_0_igdn_0_cond_1_true_202667F
Bsynthesis_layer_0_igdn_0_cond_1_identity_synthesis_layer_0_biasadd/
+synthesis_layer_0_igdn_0_cond_1_placeholder,
(synthesis_layer_0_igdn_0_cond_1_identity?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentityBsynthesis_layer_0_igdn_0_cond_1_identity_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity"]
(synthesis_layer_0_igdn_0_cond_1_identity1synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,synthesis_layer_1_igdn_1_cond_2_false_202388I
Esynthesis_layer_1_igdn_1_cond_2_cond_synthesis_layer_1_igdn_1_biasadd+
'synthesis_layer_1_igdn_1_cond_2_equal_x,
(synthesis_layer_1_igdn_1_cond_2_identity?
!synthesis/layer_1/igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_1/igdn_1/cond_2/x?
%synthesis/layer_1/igdn_1/cond_2/EqualEqual'synthesis_layer_1_igdn_1_cond_2_equal_x*synthesis/layer_1/igdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_1/igdn_1/cond_2/Equal?
$synthesis/layer_1/igdn_1/cond_2/condStatelessIf)synthesis/layer_1/igdn_1/cond_2/Equal:z:0Esynthesis_layer_1_igdn_1_cond_2_cond_synthesis_layer_1_igdn_1_biasadd'synthesis_layer_1_igdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_1_igdn_1_cond_2_cond_false_202397*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_2_cond_true_2023962&
$synthesis/layer_1/igdn_1/cond_2/cond?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity"]
(synthesis_layer_1_igdn_1_cond_2_identity1synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$igdn_1_cond_1_cond_cond_false_204432'
#igdn_1_cond_1_cond_cond_pow_biasadd!
igdn_1_cond_1_cond_cond_pow_y$
 igdn_1_cond_1_cond_cond_identity?
igdn_1/cond_1/cond/cond/powPow#igdn_1_cond_1_cond_cond_pow_biasaddigdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/cond/pow?
 igdn_1/cond_1/cond/cond/IdentityIdentityigdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity"M
 igdn_1_cond_1_cond_cond_identity)igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_2_true_204495)
%igdn_1_cond_2_identity_igdn_1_biasadd
igdn_1_cond_2_placeholder
igdn_1_cond_2_identity?
igdn_1/cond_2/IdentityIdentity%igdn_1_cond_2_identity_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity"9
igdn_1_cond_2_identityigdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
2model_synthesis_layer_1_igdn_1_cond_2_false_200290U
Qmodel_synthesis_layer_1_igdn_1_cond_2_cond_model_synthesis_layer_1_igdn_1_biasadd1
-model_synthesis_layer_1_igdn_1_cond_2_equal_x2
.model_synthesis_layer_1_igdn_1_cond_2_identity?
'model/synthesis/layer_1/igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'model/synthesis/layer_1/igdn_1/cond_2/x?
+model/synthesis/layer_1/igdn_1/cond_2/EqualEqual-model_synthesis_layer_1_igdn_1_cond_2_equal_x0model/synthesis/layer_1/igdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_1/igdn_1/cond_2/Equal?
*model/synthesis/layer_1/igdn_1/cond_2/condStatelessIf/model/synthesis/layer_1/igdn_1/cond_2/Equal:z:0Qmodel_synthesis_layer_1_igdn_1_cond_2_cond_model_synthesis_layer_1_igdn_1_biasadd-model_synthesis_layer_1_igdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_1_igdn_1_cond_2_cond_false_200299*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_1_igdn_1_cond_2_cond_true_2002982,
*model/synthesis/layer_1/igdn_1/cond_2/cond?
3model/synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity3model/synthesis/layer_1/igdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_2/cond/Identity?
.model/synthesis/layer_1/igdn_1/cond_2/IdentityIdentity<model/synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_2/Identity"i
.model_synthesis_layer_1_igdn_1_cond_2_identity7model/synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
2model_synthesis_layer_2_igdn_2_cond_1_false_200368N
Jmodel_synthesis_layer_2_igdn_2_cond_1_cond_model_synthesis_layer_2_biasadd1
-model_synthesis_layer_2_igdn_2_cond_1_equal_x2
.model_synthesis_layer_2_igdn_2_cond_1_identity?
'model/synthesis/layer_2/igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'model/synthesis/layer_2/igdn_2/cond_1/x?
+model/synthesis/layer_2/igdn_2/cond_1/EqualEqual-model_synthesis_layer_2_igdn_2_cond_1_equal_x0model/synthesis/layer_2/igdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_2/igdn_2/cond_1/Equal?
*model/synthesis/layer_2/igdn_2/cond_1/condStatelessIf/model/synthesis/layer_2/igdn_2/cond_1/Equal:z:0Jmodel_synthesis_layer_2_igdn_2_cond_1_cond_model_synthesis_layer_2_biasadd-model_synthesis_layer_2_igdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_2_igdn_2_cond_1_cond_false_200377*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_2_igdn_2_cond_1_cond_true_2003762,
*model/synthesis/layer_2/igdn_2/cond_1/cond?
3model/synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity3model/synthesis/layer_2/igdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_1/cond/Identity?
.model/synthesis/layer_2/igdn_2/cond_1/IdentityIdentity<model/synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_1/Identity"i
.model_synthesis_layer_2_igdn_2_cond_1_identity7model/synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_0_igdn_0_cond_1_cond_true_2037242
.layer_0_igdn_0_cond_1_cond_abs_layer_0_biasadd*
&layer_0_igdn_0_cond_1_cond_placeholder'
#layer_0_igdn_0_cond_1_cond_identity?
layer_0/igdn_0/cond_1/cond/AbsAbs.layer_0_igdn_0_cond_1_cond_abs_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/cond/Abs?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity"layer_0/igdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity"S
#layer_0_igdn_0_cond_1_cond_identity,layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_2_cond_1_false_204582
igdn_2_cond_1_cond_biasadd
igdn_2_cond_1_equal_x
igdn_2_cond_1_identityg
igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_2/cond_1/x?
igdn_2/cond_1/EqualEqualigdn_2_cond_1_equal_xigdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_1/Equal?
igdn_2/cond_1/condStatelessIfigdn_2/cond_1/Equal:z:0igdn_2_cond_1_cond_biasaddigdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_2_cond_1_cond_false_204591*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_1_cond_true_2045902
igdn_2/cond_1/cond?
igdn_2/cond_1/cond/IdentityIdentityigdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity?
igdn_2/cond_1/IdentityIdentity$igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity"9
igdn_2_cond_1_identityigdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
/model_synthesis_layer_1_igdn_1_cond_true_2001953
/model_synthesis_layer_1_igdn_1_cond_placeholder
0
,model_synthesis_layer_1_igdn_1_cond_identity
?
)model/synthesis/layer_1/igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)model/synthesis/layer_1/igdn_1/cond/Const?
,model/synthesis/layer_1/igdn_1/cond/IdentityIdentity2model/synthesis/layer_1/igdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_1/igdn_1/cond/Identity"e
,model_synthesis_layer_1_igdn_1_cond_identity5model/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?)
?
__inference__traced_save_204866
file_prefix+
'savev2_layer_0_bias_read_readvariableop:
6savev2_layer_0_igdn_0_reparam_beta_read_readvariableop;
7savev2_layer_0_igdn_0_reparam_gamma_read_readvariableop2
.savev2_layer_0_kernel_rdft_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop:
6savev2_layer_1_igdn_1_reparam_beta_read_readvariableop;
7savev2_layer_1_igdn_1_reparam_gamma_read_readvariableop2
.savev2_layer_1_kernel_rdft_read_readvariableop+
'savev2_layer_2_bias_read_readvariableop:
6savev2_layer_2_igdn_2_reparam_beta_read_readvariableop;
7savev2_layer_2_igdn_2_reparam_gamma_read_readvariableop2
.savev2_layer_2_kernel_rdft_read_readvariableop+
'savev2_layer_3_bias_read_readvariableop2
.savev2_layer_3_kernel_rdft_read_readvariableop
savev2_const_22

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_layer_0_bias_read_readvariableop6savev2_layer_0_igdn_0_reparam_beta_read_readvariableop7savev2_layer_0_igdn_0_reparam_gamma_read_readvariableop.savev2_layer_0_kernel_rdft_read_readvariableop'savev2_layer_1_bias_read_readvariableop6savev2_layer_1_igdn_1_reparam_beta_read_readvariableop7savev2_layer_1_igdn_1_reparam_gamma_read_readvariableop.savev2_layer_1_kernel_rdft_read_readvariableop'savev2_layer_2_bias_read_readvariableop6savev2_layer_2_igdn_2_reparam_beta_read_readvariableop7savev2_layer_2_igdn_2_reparam_gamma_read_readvariableop.savev2_layer_2_kernel_rdft_read_readvariableop'savev2_layer_3_bias_read_readvariableop.savev2_layer_3_kernel_rdft_read_readvariableopsavev2_const_22"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:
??:
??:?:?:
??:
??:?:?:
??:
??::	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!	

_output_shapes	
:?:!


_output_shapes	
:?:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??: 

_output_shapes
::%!

_output_shapes
:	?:

_output_shapes
: 
?
?
,synthesis_layer_0_igdn_0_cond_1_false_202144B
>synthesis_layer_0_igdn_0_cond_1_cond_synthesis_layer_0_biasadd+
'synthesis_layer_0_igdn_0_cond_1_equal_x,
(synthesis_layer_0_igdn_0_cond_1_identity?
!synthesis/layer_0/igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_0/igdn_0/cond_1/x?
%synthesis/layer_0/igdn_0/cond_1/EqualEqual'synthesis_layer_0_igdn_0_cond_1_equal_x*synthesis/layer_0/igdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_0/igdn_0/cond_1/Equal?
$synthesis/layer_0/igdn_0/cond_1/condStatelessIf)synthesis/layer_0/igdn_0/cond_1/Equal:z:0>synthesis_layer_0_igdn_0_cond_1_cond_synthesis_layer_0_biasadd'synthesis_layer_0_igdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_0_igdn_0_cond_1_cond_false_202153*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_1_cond_true_2021522&
$synthesis/layer_0/igdn_0/cond_1/cond?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity"]
(synthesis_layer_0_igdn_0_cond_1_identity1synthesis/layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_1_cond_1_true_200762"
igdn_1_cond_1_identity_biasadd
igdn_1_cond_1_placeholder
igdn_1_cond_1_identity?
igdn_1/cond_1/IdentityIdentityigdn_1_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity"9
igdn_1_cond_1_identityigdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
P
igdn_1_cond_true_204401
igdn_1_cond_placeholder

igdn_1_cond_identity
h
igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_1/cond/Constu
igdn_1/cond/IdentityIdentityigdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity"5
igdn_1_cond_identityigdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_203009K
Gsynthesis_layer_2_igdn_2_cond_1_cond_cond_pow_synthesis_layer_2_biasadd3
/synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y6
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
-synthesis/layer_2/igdn_2/cond_1/cond/cond/powPowGsynthesis_layer_2_igdn_2_cond_1_cond_cond_pow_synthesis_layer_2_biasadd/synthesis_layer_2_igdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/cond/pow?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity1synthesis/layer_2/igdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"q
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_0_igdn_0_cond_1_false_203192.
*layer_0_igdn_0_cond_1_cond_layer_0_biasadd!
layer_0_igdn_0_cond_1_equal_x"
layer_0_igdn_0_cond_1_identityw
layer_0/igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/cond_1/x?
layer_0/igdn_0/cond_1/EqualEquallayer_0_igdn_0_cond_1_equal_x layer_0/igdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/cond_1/Equal?
layer_0/igdn_0/cond_1/condStatelessIflayer_0/igdn_0/cond_1/Equal:z:0*layer_0_igdn_0_cond_1_cond_layer_0_biasaddlayer_0_igdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_0_igdn_0_cond_1_cond_false_203201*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_1_cond_true_2032002
layer_0/igdn_0/cond_1/cond?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity#layer_0/igdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity?
layer_0/igdn_0/cond_1/IdentityIdentity,layer_0/igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity"I
layer_0_igdn_0_cond_1_identity'layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_0_cond_1_cond_true_200582"
igdn_0_cond_1_cond_abs_biasadd"
igdn_0_cond_1_cond_placeholder
igdn_0_cond_1_cond_identity?
igdn_0/cond_1/cond/AbsAbsigdn_0_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Abs?
igdn_0/cond_1/cond/IdentityIdentityigdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity"C
igdn_0_cond_1_cond_identity$igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_0_igdn_0_cond_1_false_203716.
*layer_0_igdn_0_cond_1_cond_layer_0_biasadd!
layer_0_igdn_0_cond_1_equal_x"
layer_0_igdn_0_cond_1_identityw
layer_0/igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/cond_1/x?
layer_0/igdn_0/cond_1/EqualEquallayer_0_igdn_0_cond_1_equal_x layer_0/igdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/cond_1/Equal?
layer_0/igdn_0/cond_1/condStatelessIflayer_0/igdn_0/cond_1/Equal:z:0*layer_0_igdn_0_cond_1_cond_layer_0_biasaddlayer_0_igdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_0_igdn_0_cond_1_cond_false_203725*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_1_cond_true_2037242
layer_0/igdn_0/cond_1/cond?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity#layer_0/igdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity?
layer_0/igdn_0/cond_1/IdentityIdentity,layer_0/igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity"I
layer_0_igdn_0_cond_1_identity'layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
2model_synthesis_layer_0_igdn_0_cond_2_false_200129U
Qmodel_synthesis_layer_0_igdn_0_cond_2_cond_model_synthesis_layer_0_igdn_0_biasadd1
-model_synthesis_layer_0_igdn_0_cond_2_equal_x2
.model_synthesis_layer_0_igdn_0_cond_2_identity?
'model/synthesis/layer_0/igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'model/synthesis/layer_0/igdn_0/cond_2/x?
+model/synthesis/layer_0/igdn_0/cond_2/EqualEqual-model_synthesis_layer_0_igdn_0_cond_2_equal_x0model/synthesis/layer_0/igdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_0/igdn_0/cond_2/Equal?
*model/synthesis/layer_0/igdn_0/cond_2/condStatelessIf/model/synthesis/layer_0/igdn_0/cond_2/Equal:z:0Qmodel_synthesis_layer_0_igdn_0_cond_2_cond_model_synthesis_layer_0_igdn_0_biasadd-model_synthesis_layer_0_igdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_0_igdn_0_cond_2_cond_false_200138*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_0_igdn_0_cond_2_cond_true_2001372,
*model/synthesis/layer_0/igdn_0/cond_2/cond?
3model/synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity3model/synthesis/layer_0/igdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_2/cond/Identity?
.model/synthesis/layer_0/igdn_0/cond_2/IdentityIdentity<model/synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_2/Identity"i
.model_synthesis_layer_0_igdn_0_cond_2_identity7model/synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_0_igdn_0_cond_2_cond_true_202235N
Jsynthesis_layer_0_igdn_0_cond_2_cond_sqrt_synthesis_layer_0_igdn_0_biasadd4
0synthesis_layer_0_igdn_0_cond_2_cond_placeholder1
-synthesis_layer_0_igdn_0_cond_2_cond_identity?
)synthesis/layer_0/igdn_0/cond_2/cond/SqrtSqrtJsynthesis_layer_0_igdn_0_cond_2_cond_sqrt_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_0/igdn_0/cond_2/cond/Sqrt?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_2_cond_identity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_2_igdn_2_cond_1_true_2035132
.layer_2_igdn_2_cond_1_identity_layer_2_biasadd%
!layer_2_igdn_2_cond_1_placeholder"
layer_2_igdn_2_cond_1_identity?
layer_2/igdn_2/cond_1/IdentityIdentity.layer_2_igdn_2_cond_1_identity_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity"I
layer_2_igdn_2_cond_1_identity'layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
{
 layer_1_igdn_1_cond_false_2038665
1layer_1_igdn_1_cond_identity_layer_1_igdn_1_equal
 
layer_1_igdn_1_cond_identity
?
layer_1/igdn_1/cond/IdentityIdentity1layer_1_igdn_1_cond_identity_layer_1_igdn_1_equal*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity"E
layer_1_igdn_1_cond_identity%layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1synthesis_layer_0_igdn_0_cond_1_cond_false_202677G
Csynthesis_layer_0_igdn_0_cond_1_cond_cond_synthesis_layer_0_biasadd0
,synthesis_layer_0_igdn_0_cond_1_cond_equal_x1
-synthesis_layer_0_igdn_0_cond_1_cond_identity?
&synthesis/layer_0/igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2(
&synthesis/layer_0/igdn_0/cond_1/cond/x?
*synthesis/layer_0/igdn_0/cond_1/cond/EqualEqual,synthesis_layer_0_igdn_0_cond_1_cond_equal_x/synthesis/layer_0/igdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2,
*synthesis/layer_0/igdn_0/cond_1/cond/Equal?
)synthesis/layer_0/igdn_0/cond_1/cond/condStatelessIf.synthesis/layer_0/igdn_0/cond_1/cond/Equal:z:0Csynthesis_layer_0_igdn_0_cond_1_cond_cond_synthesis_layer_0_biasadd,synthesis_layer_0_igdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *I
else_branch:R8
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202687*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_2026862+
)synthesis/layer_0/igdn_0/cond_1/cond/cond?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity2synthesis/layer_0/igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity?
-synthesis/layer_0/igdn_0/cond_1/cond/IdentityIdentity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/Identity"g
-synthesis_layer_0_igdn_0_cond_1_cond_identity6synthesis/layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_2_cond_1_cond_cond_true_200970*
&igdn_2_cond_1_cond_cond_square_biasadd'
#igdn_2_cond_1_cond_cond_placeholder$
 igdn_2_cond_1_cond_cond_identity?
igdn_2/cond_1/cond/cond/SquareSquare&igdn_2_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_2/cond_1/cond/cond/Square?
 igdn_2/cond_1/cond/cond/IdentityIdentity"igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity"M
 igdn_2_cond_1_cond_cond_identity)igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
2model_synthesis_layer_1_igdn_1_cond_1_false_200207N
Jmodel_synthesis_layer_1_igdn_1_cond_1_cond_model_synthesis_layer_1_biasadd1
-model_synthesis_layer_1_igdn_1_cond_1_equal_x2
.model_synthesis_layer_1_igdn_1_cond_1_identity?
'model/synthesis/layer_1/igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'model/synthesis/layer_1/igdn_1/cond_1/x?
+model/synthesis/layer_1/igdn_1/cond_1/EqualEqual-model_synthesis_layer_1_igdn_1_cond_1_equal_x0model/synthesis/layer_1/igdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+model/synthesis/layer_1/igdn_1/cond_1/Equal?
*model/synthesis/layer_1/igdn_1/cond_1/condStatelessIf/model/synthesis/layer_1/igdn_1/cond_1/Equal:z:0Jmodel_synthesis_layer_1_igdn_1_cond_1_cond_model_synthesis_layer_1_biasadd-model_synthesis_layer_1_igdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *J
else_branch;R9
7model_synthesis_layer_1_igdn_1_cond_1_cond_false_200216*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_1_igdn_1_cond_1_cond_true_2002152,
*model/synthesis/layer_1/igdn_1/cond_1/cond?
3model/synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity3model/synthesis/layer_1/igdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_1/cond/Identity?
.model/synthesis/layer_1/igdn_1/cond_1/IdentityIdentity<model/synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_1/Identity"i
.model_synthesis_layer_1_igdn_1_cond_1_identity7model/synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_2_igdn_2_cond_1_cond_true_202474F
Bsynthesis_layer_2_igdn_2_cond_1_cond_abs_synthesis_layer_2_biasadd4
0synthesis_layer_2_igdn_2_cond_1_cond_placeholder1
-synthesis_layer_2_igdn_2_cond_1_cond_identity?
(synthesis/layer_2/igdn_2/cond_1/cond/AbsAbsBsynthesis_layer_2_igdn_2_cond_1_cond_abs_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/cond/Abs?
-synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity,synthesis/layer_2/igdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_2/igdn_2/cond_1/cond/Identity"g
-synthesis_layer_2_igdn_2_cond_1_cond_identity6synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
7model_synthesis_layer_1_igdn_1_cond_1_cond_false_200216S
Omodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_model_synthesis_layer_1_biasadd6
2model_synthesis_layer_1_igdn_1_cond_1_cond_equal_x7
3model_synthesis_layer_1_igdn_1_cond_1_cond_identity?
,model/synthesis/layer_1/igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,model/synthesis/layer_1/igdn_1/cond_1/cond/x?
0model/synthesis/layer_1/igdn_1/cond_1/cond/EqualEqual2model_synthesis_layer_1_igdn_1_cond_1_cond_equal_x5model/synthesis/layer_1/igdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0model/synthesis/layer_1/igdn_1/cond_1/cond/Equal?
/model/synthesis/layer_1/igdn_1/cond_1/cond/condStatelessIf4model/synthesis/layer_1/igdn_1/cond_1/cond/Equal:z:0Omodel_synthesis_layer_1_igdn_1_cond_1_cond_cond_model_synthesis_layer_1_biasadd2model_synthesis_layer_1_igdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *O
else_branch@R>
<model_synthesis_layer_1_igdn_1_cond_1_cond_cond_false_200226*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_1_igdn_1_cond_1_cond_cond_true_20022521
/model/synthesis/layer_1/igdn_1/cond_1/cond/cond?
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity8model/synthesis/layer_1/igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity?
3model/synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentityAmodel/synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_1/igdn_1/cond_1/cond/Identity"s
3model_synthesis_layer_1_igdn_1_cond_1_cond_identity<model/synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$igdn_2_cond_1_cond_cond_false_200971'
#igdn_2_cond_1_cond_cond_pow_biasadd!
igdn_2_cond_1_cond_cond_pow_y$
 igdn_2_cond_1_cond_cond_identity?
igdn_2/cond_1/cond/cond/powPow#igdn_2_cond_1_cond_cond_pow_biasaddigdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/cond/pow?
 igdn_2/cond_1/cond/cond/IdentityIdentityigdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_2/cond_1/cond/cond/Identity"M
 igdn_2_cond_1_cond_cond_identity)igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_202162N
Jsynthesis_layer_0_igdn_0_cond_1_cond_cond_square_synthesis_layer_0_biasadd9
5synthesis_layer_0_igdn_0_cond_1_cond_cond_placeholder6
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
0synthesis/layer_0/igdn_0/cond_1/cond/cond/SquareSquareJsynthesis_layer_0_igdn_0_cond_1_cond_cond_square_synthesis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_0/igdn_0/cond_1/cond/cond/Square?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity4synthesis/layer_0/igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"q
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_1_igdn_1_cond_1_cond_true_2033612
.layer_1_igdn_1_cond_1_cond_abs_layer_1_biasadd*
&layer_1_igdn_1_cond_1_cond_placeholder'
#layer_1_igdn_1_cond_1_cond_identity?
layer_1/igdn_1/cond_1/cond/AbsAbs.layer_1_igdn_1_cond_1_cond_abs_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/cond/Abs?
#layer_1/igdn_1/cond_1/cond/IdentityIdentity"layer_1/igdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/Identity"S
#layer_1_igdn_1_cond_1_cond_identity,layer_1/igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_1_cond_2_false_204496%
!igdn_1_cond_2_cond_igdn_1_biasadd
igdn_1_cond_2_equal_x
igdn_1_cond_2_identityg
igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_1/cond_2/x?
igdn_1/cond_2/EqualEqualigdn_1_cond_2_equal_xigdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_2/Equal?
igdn_1/cond_2/condStatelessIfigdn_1/cond_2/Equal:z:0!igdn_1_cond_2_cond_igdn_1_biasaddigdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_1_cond_2_cond_false_204505*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_2_cond_true_2045042
igdn_1/cond_2/cond?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity?
igdn_1/cond_2/IdentityIdentity$igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity"9
igdn_1_cond_2_identityigdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202848K
Gsynthesis_layer_1_igdn_1_cond_1_cond_cond_pow_synthesis_layer_1_biasadd3
/synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y6
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity?
-synthesis/layer_1/igdn_1/cond_1/cond/cond/powPowGsynthesis_layer_1_igdn_1_cond_1_cond_cond_pow_synthesis_layer_1_biasadd/synthesis_layer_1_igdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/cond/pow?
2synthesis/layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity1synthesis/layer_1/igdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity"q
2synthesis_layer_1_igdn_1_cond_1_cond_cond_identity;synthesis/layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
P
igdn_0_cond_true_200562
igdn_0_cond_placeholder

igdn_0_cond_identity
h
igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_0/cond/Constu
igdn_0/cond/IdentityIdentityigdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity"5
igdn_0_cond_identityigdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
'layer_2_igdn_2_cond_2_cond_false_2041309
5layer_2_igdn_2_cond_2_cond_pow_layer_2_igdn_2_biasadd$
 layer_2_igdn_2_cond_2_cond_pow_y'
#layer_2_igdn_2_cond_2_cond_identity?
layer_2/igdn_2/cond_2/cond/powPow5layer_2_igdn_2_cond_2_cond_pow_layer_2_igdn_2_biasadd layer_2_igdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/cond/pow?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity"layer_2/igdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity"S
#layer_2_igdn_2_cond_2_cond_identity,layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_1_cond_false_200752%
!igdn_1_cond_identity_igdn_1_equal

igdn_1_cond_identity
|
igdn_1/cond/IdentityIdentity!igdn_1_cond_identity_igdn_1_equal*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity"5
igdn_1_cond_identityigdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1model_synthesis_layer_2_igdn_2_cond_1_true_200367R
Nmodel_synthesis_layer_2_igdn_2_cond_1_identity_model_synthesis_layer_2_biasadd5
1model_synthesis_layer_2_igdn_2_cond_1_placeholder2
.model_synthesis_layer_2_igdn_2_cond_1_identity?
.model/synthesis/layer_2/igdn_2/cond_1/IdentityIdentityNmodel_synthesis_layer_2_igdn_2_cond_1_identity_model_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_1/Identity"i
.model_synthesis_layer_2_igdn_2_cond_1_identity7model/synthesis/layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0model_synthesis_layer_0_igdn_0_cond_false_200035U
Qmodel_synthesis_layer_0_igdn_0_cond_identity_model_synthesis_layer_0_igdn_0_equal
0
,model_synthesis_layer_0_igdn_0_cond_identity
?
,model/synthesis/layer_0/igdn_0/cond/IdentityIdentityQmodel_synthesis_layer_0_igdn_0_cond_identity_model_synthesis_layer_0_igdn_0_equal*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_0/igdn_0/cond/Identity"e
,model_synthesis_layer_0_igdn_0_cond_identity5model/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
igdn_2_cond_1_cond_true_204590"
igdn_2_cond_1_cond_abs_biasadd"
igdn_2_cond_1_cond_placeholder
igdn_2_cond_1_cond_identity?
igdn_2/cond_1/cond/AbsAbsigdn_2_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Abs?
igdn_2/cond_1/cond/IdentityIdentityigdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity"C
igdn_2_cond_1_cond_identity$igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_0_igdn_0_cond_1_cond_false_2032013
/layer_0_igdn_0_cond_1_cond_cond_layer_0_biasadd&
"layer_0_igdn_0_cond_1_cond_equal_x'
#layer_0_igdn_0_cond_1_cond_identity?
layer_0/igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_0/igdn_0/cond_1/cond/x?
 layer_0/igdn_0/cond_1/cond/EqualEqual"layer_0_igdn_0_cond_1_cond_equal_x%layer_0/igdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 layer_0/igdn_0/cond_1/cond/Equal?
layer_0/igdn_0/cond_1/cond/condStatelessIf$layer_0/igdn_0/cond_1/cond/Equal:z:0/layer_0_igdn_0_cond_1_cond_cond_layer_0_biasadd"layer_0_igdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,layer_0_igdn_0_cond_1_cond_cond_false_203211*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_0_igdn_0_cond_1_cond_cond_true_2032102!
layer_0/igdn_0/cond_1/cond/cond?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity(layer_0/igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity?
#layer_0/igdn_0/cond_1/cond/IdentityIdentity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/Identity"S
#layer_0_igdn_0_cond_1_cond_identity,layer_0/igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_202484N
Jsynthesis_layer_2_igdn_2_cond_1_cond_cond_square_synthesis_layer_2_biasadd9
5synthesis_layer_2_igdn_2_cond_1_cond_cond_placeholder6
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity?
0synthesis/layer_2/igdn_2/cond_1/cond/cond/SquareSquareJsynthesis_layer_2_igdn_2_cond_1_cond_cond_square_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????22
0synthesis/layer_2/igdn_2/cond_1/cond/cond/Square?
2synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity4synthesis/layer_2/igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity"q
2synthesis_layer_2_igdn_2_cond_1_cond_cond_identity;synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
z
igdn_2_cond_2_false_204665%
!igdn_2_cond_2_cond_igdn_2_biasadd
igdn_2_cond_2_equal_x
igdn_2_cond_2_identityg
igdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
igdn_2/cond_2/x?
igdn_2/cond_2/EqualEqualigdn_2_cond_2_equal_xigdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_2/Equal?
igdn_2/cond_2/condStatelessIfigdn_2/cond_2/Equal:z:0!igdn_2_cond_2_cond_igdn_2_biasaddigdn_2_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_2_cond_2_cond_false_204674*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_2_cond_true_2046732
igdn_2/cond_2/cond?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity?
igdn_2/cond_2/IdentityIdentity$igdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/Identity"9
igdn_2_cond_2_identityigdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
??
?
!__inference__wrapped_model_200517
input_2
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??F
7model_synthesis_layer_0_biasadd_readvariableop_resource:	?*
&model_synthesis_layer_0_igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y,
(model_synthesis_layer_0_igdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??F
7model_synthesis_layer_1_biasadd_readvariableop_resource:	?*
&model_synthesis_layer_1_igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y,
(model_synthesis_layer_1_igdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??F
7model_synthesis_layer_2_biasadd_readvariableop_resource:	?*
&model_synthesis_layer_2_igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y,
(model_synthesis_layer_2_igdn_2_equal_1_x
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	?E
7model_synthesis_layer_3_biasadd_readvariableop_resource:
identity??.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?.model/synthesis/layer_0/BiasAdd/ReadVariableOp?.model/synthesis/layer_1/BiasAdd/ReadVariableOp?.model/synthesis/layer_2/BiasAdd/ReadVariableOp?.model/synthesis/layer_3/BiasAdd/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshape?
&model/synthesis/layer_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/synthesis/layer_0/transpose/perm?
!model/synthesis/layer_0/transpose	Transposelayer_0/kernel/Reshape:output:0/model/synthesis/layer_0/transpose/perm:output:0*
T0*(
_output_shapes
:??2#
!model/synthesis/layer_0/transposeu
model/synthesis/layer_0/ShapeShapeinput_2*
T0*
_output_shapes
:2
model/synthesis/layer_0/Shape?
+model/synthesis/layer_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/synthesis/layer_0/strided_slice/stack?
-model/synthesis/layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_0/strided_slice/stack_1?
-model/synthesis/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_0/strided_slice/stack_2?
%model/synthesis/layer_0/strided_sliceStridedSlice&model/synthesis/layer_0/Shape:output:04model/synthesis/layer_0/strided_slice/stack:output:06model/synthesis/layer_0/strided_slice/stack_1:output:06model/synthesis/layer_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/synthesis/layer_0/strided_slice?
-model/synthesis/layer_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_0/strided_slice_1/stack?
/model/synthesis/layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_0/strided_slice_1/stack_1?
/model/synthesis/layer_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_0/strided_slice_1/stack_2?
'model/synthesis/layer_0/strided_slice_1StridedSlice&model/synthesis/layer_0/Shape:output:06model/synthesis/layer_0/strided_slice_1/stack:output:08model/synthesis/layer_0/strided_slice_1/stack_1:output:08model/synthesis/layer_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_0/strided_slice_1?
model/synthesis/layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/synthesis/layer_0/mul/y?
model/synthesis/layer_0/mulMul0model/synthesis/layer_0/strided_slice_1:output:0&model/synthesis/layer_0/mul/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_0/mul?
model/synthesis/layer_0/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/synthesis/layer_0/add/y?
model/synthesis/layer_0/addAddV2model/synthesis/layer_0/mul:z:0&model/synthesis/layer_0/add/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_0/add?
-model/synthesis/layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_0/strided_slice_2/stack?
/model/synthesis/layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_0/strided_slice_2/stack_1?
/model/synthesis/layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_0/strided_slice_2/stack_2?
'model/synthesis/layer_0/strided_slice_2StridedSlice&model/synthesis/layer_0/Shape:output:06model/synthesis/layer_0/strided_slice_2/stack:output:08model/synthesis/layer_0/strided_slice_2/stack_1:output:08model/synthesis/layer_0/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_0/strided_slice_2?
model/synthesis/layer_0/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/synthesis/layer_0/mul_1/y?
model/synthesis/layer_0/mul_1Mul0model/synthesis/layer_0/strided_slice_2:output:0(model/synthesis/layer_0/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_0/mul_1?
model/synthesis/layer_0/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
model/synthesis/layer_0/add_1/y?
model/synthesis/layer_0/add_1AddV2!model/synthesis/layer_0/mul_1:z:0(model/synthesis/layer_0/add_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_0/add_1?
6model/synthesis/layer_0/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?28
6model/synthesis/layer_0/conv2d_transpose/input_sizes/3?
4model/synthesis/layer_0/conv2d_transpose/input_sizesPack.model/synthesis/layer_0/strided_slice:output:0model/synthesis/layer_0/add:z:0!model/synthesis/layer_0/add_1:z:0?model/synthesis/layer_0/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:26
4model/synthesis/layer_0/conv2d_transpose/input_sizes?
(model/synthesis/layer_0/conv2d_transposeConv2DBackpropInput=model/synthesis/layer_0/conv2d_transpose/input_sizes:output:0%model/synthesis/layer_0/transpose:y:0input_2*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(model/synthesis/layer_0/conv2d_transpose?
.model/synthesis/layer_0/BiasAdd/ReadVariableOpReadVariableOp7model_synthesis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/synthesis/layer_0/BiasAdd/ReadVariableOp?
model/synthesis/layer_0/BiasAddBiasAdd1model/synthesis/layer_0/conv2d_transpose:output:06model/synthesis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
model/synthesis/layer_0/BiasAdd?
 model/synthesis/layer_0/igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 model/synthesis/layer_0/igdn_0/x?
$model/synthesis/layer_0/igdn_0/EqualEqual&model_synthesis_layer_0_igdn_0_equal_x)model/synthesis/layer_0/igdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$model/synthesis/layer_0/igdn_0/Equal?
#model/synthesis/layer_0/igdn_0/condStatelessIf(model/synthesis/layer_0/igdn_0/Equal:z:0(model/synthesis/layer_0/igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *C
else_branch4R2
0model_synthesis_layer_0_igdn_0_cond_false_200035*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_0_igdn_0_cond_true_2000342%
#model/synthesis/layer_0/igdn_0/cond?
,model/synthesis/layer_0/igdn_0/cond/IdentityIdentity,model/synthesis/layer_0/igdn_0/cond:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_0/igdn_0/cond/Identity?
%model/synthesis/layer_0/igdn_0/cond_1StatelessIf5model/synthesis/layer_0/igdn_0/cond/Identity:output:0(model/synthesis/layer_0/BiasAdd:output:0&model_synthesis_layer_0_igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_0_igdn_0_cond_1_false_200046*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_0_igdn_0_cond_1_true_2000452'
%model/synthesis/layer_0/igdn_0/cond_1?
.model/synthesis/layer_0/igdn_0/cond_1/IdentityIdentity.model/synthesis/layer_0/igdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200091*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200101*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
,model/synthesis/layer_0/igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,model/synthesis/layer_0/igdn_0/Reshape/shape?
&model/synthesis/layer_0/igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:05model/synthesis/layer_0/igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&model/synthesis/layer_0/igdn_0/Reshape?
*model/synthesis/layer_0/igdn_0/convolutionConv2D7model/synthesis/layer_0/igdn_0/cond_1/Identity:output:0/model/synthesis/layer_0/igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*model/synthesis/layer_0/igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200115*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
&model/synthesis/layer_0/igdn_0/BiasAddBiasAdd3model/synthesis/layer_0/igdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&model/synthesis/layer_0/igdn_0/BiasAdd?
"model/synthesis/layer_0/igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"model/synthesis/layer_0/igdn_0/x_1?
&model/synthesis/layer_0/igdn_0/Equal_1Equal(model_synthesis_layer_0_igdn_0_equal_1_x+model/synthesis/layer_0/igdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&model/synthesis/layer_0/igdn_0/Equal_1?
%model/synthesis/layer_0/igdn_0/cond_2StatelessIf*model/synthesis/layer_0/igdn_0/Equal_1:z:0/model/synthesis/layer_0/igdn_0/BiasAdd:output:0(model_synthesis_layer_0_igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_0_igdn_0_cond_2_false_200129*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_0_igdn_0_cond_2_true_2001282'
%model/synthesis/layer_0/igdn_0/cond_2?
.model/synthesis/layer_0/igdn_0/cond_2/IdentityIdentity.model/synthesis/layer_0/igdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_0/igdn_0/cond_2/Identity?
"model/synthesis/layer_0/igdn_0/mulMul(model/synthesis/layer_0/BiasAdd:output:07model/synthesis/layer_0/igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"model/synthesis/layer_0/igdn_0/mul?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshape?
&model/synthesis/layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/synthesis/layer_1/transpose/perm?
!model/synthesis/layer_1/transpose	Transposelayer_1/kernel/Reshape:output:0/model/synthesis/layer_1/transpose/perm:output:0*
T0*(
_output_shapes
:??2#
!model/synthesis/layer_1/transpose?
model/synthesis/layer_1/ShapeShape&model/synthesis/layer_0/igdn_0/mul:z:0*
T0*
_output_shapes
:2
model/synthesis/layer_1/Shape?
+model/synthesis/layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/synthesis/layer_1/strided_slice/stack?
-model/synthesis/layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_1/strided_slice/stack_1?
-model/synthesis/layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_1/strided_slice/stack_2?
%model/synthesis/layer_1/strided_sliceStridedSlice&model/synthesis/layer_1/Shape:output:04model/synthesis/layer_1/strided_slice/stack:output:06model/synthesis/layer_1/strided_slice/stack_1:output:06model/synthesis/layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/synthesis/layer_1/strided_slice?
-model/synthesis/layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_1/strided_slice_1/stack?
/model/synthesis/layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_1/strided_slice_1/stack_1?
/model/synthesis/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_1/strided_slice_1/stack_2?
'model/synthesis/layer_1/strided_slice_1StridedSlice&model/synthesis/layer_1/Shape:output:06model/synthesis/layer_1/strided_slice_1/stack:output:08model/synthesis/layer_1/strided_slice_1/stack_1:output:08model/synthesis/layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_1/strided_slice_1?
model/synthesis/layer_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/synthesis/layer_1/mul/y?
model/synthesis/layer_1/mulMul0model/synthesis/layer_1/strided_slice_1:output:0&model/synthesis/layer_1/mul/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_1/mul?
model/synthesis/layer_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/synthesis/layer_1/add/y?
model/synthesis/layer_1/addAddV2model/synthesis/layer_1/mul:z:0&model/synthesis/layer_1/add/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_1/add?
-model/synthesis/layer_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_1/strided_slice_2/stack?
/model/synthesis/layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_1/strided_slice_2/stack_1?
/model/synthesis/layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_1/strided_slice_2/stack_2?
'model/synthesis/layer_1/strided_slice_2StridedSlice&model/synthesis/layer_1/Shape:output:06model/synthesis/layer_1/strided_slice_2/stack:output:08model/synthesis/layer_1/strided_slice_2/stack_1:output:08model/synthesis/layer_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_1/strided_slice_2?
model/synthesis/layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/synthesis/layer_1/mul_1/y?
model/synthesis/layer_1/mul_1Mul0model/synthesis/layer_1/strided_slice_2:output:0(model/synthesis/layer_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_1/mul_1?
model/synthesis/layer_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
model/synthesis/layer_1/add_1/y?
model/synthesis/layer_1/add_1AddV2!model/synthesis/layer_1/mul_1:z:0(model/synthesis/layer_1/add_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_1/add_1?
6model/synthesis/layer_1/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?28
6model/synthesis/layer_1/conv2d_transpose/input_sizes/3?
4model/synthesis/layer_1/conv2d_transpose/input_sizesPack.model/synthesis/layer_1/strided_slice:output:0model/synthesis/layer_1/add:z:0!model/synthesis/layer_1/add_1:z:0?model/synthesis/layer_1/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:26
4model/synthesis/layer_1/conv2d_transpose/input_sizes?
(model/synthesis/layer_1/conv2d_transposeConv2DBackpropInput=model/synthesis/layer_1/conv2d_transpose/input_sizes:output:0%model/synthesis/layer_1/transpose:y:0&model/synthesis/layer_0/igdn_0/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(model/synthesis/layer_1/conv2d_transpose?
.model/synthesis/layer_1/BiasAdd/ReadVariableOpReadVariableOp7model_synthesis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/synthesis/layer_1/BiasAdd/ReadVariableOp?
model/synthesis/layer_1/BiasAddBiasAdd1model/synthesis/layer_1/conv2d_transpose:output:06model/synthesis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
model/synthesis/layer_1/BiasAdd?
 model/synthesis/layer_1/igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 model/synthesis/layer_1/igdn_1/x?
$model/synthesis/layer_1/igdn_1/EqualEqual&model_synthesis_layer_1_igdn_1_equal_x)model/synthesis/layer_1/igdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$model/synthesis/layer_1/igdn_1/Equal?
#model/synthesis/layer_1/igdn_1/condStatelessIf(model/synthesis/layer_1/igdn_1/Equal:z:0(model/synthesis/layer_1/igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *C
else_branch4R2
0model_synthesis_layer_1_igdn_1_cond_false_200196*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_1_igdn_1_cond_true_2001952%
#model/synthesis/layer_1/igdn_1/cond?
,model/synthesis/layer_1/igdn_1/cond/IdentityIdentity,model/synthesis/layer_1/igdn_1/cond:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_1/igdn_1/cond/Identity?
%model/synthesis/layer_1/igdn_1/cond_1StatelessIf5model/synthesis/layer_1/igdn_1/cond/Identity:output:0(model/synthesis/layer_1/BiasAdd:output:0&model_synthesis_layer_1_igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_1_igdn_1_cond_1_false_200207*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_1_igdn_1_cond_1_true_2002062'
%model/synthesis/layer_1/igdn_1/cond_1?
.model/synthesis/layer_1/igdn_1/cond_1/IdentityIdentity.model/synthesis/layer_1/igdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200252*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200262*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
,model/synthesis/layer_1/igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,model/synthesis/layer_1/igdn_1/Reshape/shape?
&model/synthesis/layer_1/igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:05model/synthesis/layer_1/igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&model/synthesis/layer_1/igdn_1/Reshape?
*model/synthesis/layer_1/igdn_1/convolutionConv2D7model/synthesis/layer_1/igdn_1/cond_1/Identity:output:0/model/synthesis/layer_1/igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*model/synthesis/layer_1/igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200276*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
&model/synthesis/layer_1/igdn_1/BiasAddBiasAdd3model/synthesis/layer_1/igdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&model/synthesis/layer_1/igdn_1/BiasAdd?
"model/synthesis/layer_1/igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"model/synthesis/layer_1/igdn_1/x_1?
&model/synthesis/layer_1/igdn_1/Equal_1Equal(model_synthesis_layer_1_igdn_1_equal_1_x+model/synthesis/layer_1/igdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&model/synthesis/layer_1/igdn_1/Equal_1?
%model/synthesis/layer_1/igdn_1/cond_2StatelessIf*model/synthesis/layer_1/igdn_1/Equal_1:z:0/model/synthesis/layer_1/igdn_1/BiasAdd:output:0(model_synthesis_layer_1_igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_1_igdn_1_cond_2_false_200290*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_1_igdn_1_cond_2_true_2002892'
%model/synthesis/layer_1/igdn_1/cond_2?
.model/synthesis/layer_1/igdn_1/cond_2/IdentityIdentity.model/synthesis/layer_1/igdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_2/Identity?
"model/synthesis/layer_1/igdn_1/mulMul(model/synthesis/layer_1/BiasAdd:output:07model/synthesis/layer_1/igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"model/synthesis/layer_1/igdn_1/mul?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshape?
&model/synthesis/layer_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/synthesis/layer_2/transpose/perm?
!model/synthesis/layer_2/transpose	Transposelayer_2/kernel/Reshape:output:0/model/synthesis/layer_2/transpose/perm:output:0*
T0*(
_output_shapes
:??2#
!model/synthesis/layer_2/transpose?
model/synthesis/layer_2/ShapeShape&model/synthesis/layer_1/igdn_1/mul:z:0*
T0*
_output_shapes
:2
model/synthesis/layer_2/Shape?
+model/synthesis/layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/synthesis/layer_2/strided_slice/stack?
-model/synthesis/layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_2/strided_slice/stack_1?
-model/synthesis/layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_2/strided_slice/stack_2?
%model/synthesis/layer_2/strided_sliceStridedSlice&model/synthesis/layer_2/Shape:output:04model/synthesis/layer_2/strided_slice/stack:output:06model/synthesis/layer_2/strided_slice/stack_1:output:06model/synthesis/layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/synthesis/layer_2/strided_slice?
-model/synthesis/layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_2/strided_slice_1/stack?
/model/synthesis/layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_2/strided_slice_1/stack_1?
/model/synthesis/layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_2/strided_slice_1/stack_2?
'model/synthesis/layer_2/strided_slice_1StridedSlice&model/synthesis/layer_2/Shape:output:06model/synthesis/layer_2/strided_slice_1/stack:output:08model/synthesis/layer_2/strided_slice_1/stack_1:output:08model/synthesis/layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_2/strided_slice_1?
model/synthesis/layer_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/synthesis/layer_2/mul/y?
model/synthesis/layer_2/mulMul0model/synthesis/layer_2/strided_slice_1:output:0&model/synthesis/layer_2/mul/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_2/mul?
model/synthesis/layer_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/synthesis/layer_2/add/y?
model/synthesis/layer_2/addAddV2model/synthesis/layer_2/mul:z:0&model/synthesis/layer_2/add/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_2/add?
-model/synthesis/layer_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_2/strided_slice_2/stack?
/model/synthesis/layer_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_2/strided_slice_2/stack_1?
/model/synthesis/layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_2/strided_slice_2/stack_2?
'model/synthesis/layer_2/strided_slice_2StridedSlice&model/synthesis/layer_2/Shape:output:06model/synthesis/layer_2/strided_slice_2/stack:output:08model/synthesis/layer_2/strided_slice_2/stack_1:output:08model/synthesis/layer_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_2/strided_slice_2?
model/synthesis/layer_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/synthesis/layer_2/mul_1/y?
model/synthesis/layer_2/mul_1Mul0model/synthesis/layer_2/strided_slice_2:output:0(model/synthesis/layer_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_2/mul_1?
model/synthesis/layer_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
model/synthesis/layer_2/add_1/y?
model/synthesis/layer_2/add_1AddV2!model/synthesis/layer_2/mul_1:z:0(model/synthesis/layer_2/add_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_2/add_1?
6model/synthesis/layer_2/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?28
6model/synthesis/layer_2/conv2d_transpose/input_sizes/3?
4model/synthesis/layer_2/conv2d_transpose/input_sizesPack.model/synthesis/layer_2/strided_slice:output:0model/synthesis/layer_2/add:z:0!model/synthesis/layer_2/add_1:z:0?model/synthesis/layer_2/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:26
4model/synthesis/layer_2/conv2d_transpose/input_sizes?
(model/synthesis/layer_2/conv2d_transposeConv2DBackpropInput=model/synthesis/layer_2/conv2d_transpose/input_sizes:output:0%model/synthesis/layer_2/transpose:y:0&model/synthesis/layer_1/igdn_1/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(model/synthesis/layer_2/conv2d_transpose?
.model/synthesis/layer_2/BiasAdd/ReadVariableOpReadVariableOp7model_synthesis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype020
.model/synthesis/layer_2/BiasAdd/ReadVariableOp?
model/synthesis/layer_2/BiasAddBiasAdd1model/synthesis/layer_2/conv2d_transpose:output:06model/synthesis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
model/synthesis/layer_2/BiasAdd?
 model/synthesis/layer_2/igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 model/synthesis/layer_2/igdn_2/x?
$model/synthesis/layer_2/igdn_2/EqualEqual&model_synthesis_layer_2_igdn_2_equal_x)model/synthesis/layer_2/igdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$model/synthesis/layer_2/igdn_2/Equal?
#model/synthesis/layer_2/igdn_2/condStatelessIf(model/synthesis/layer_2/igdn_2/Equal:z:0(model/synthesis/layer_2/igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *C
else_branch4R2
0model_synthesis_layer_2_igdn_2_cond_false_200357*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_2_igdn_2_cond_true_2003562%
#model/synthesis/layer_2/igdn_2/cond?
,model/synthesis/layer_2/igdn_2/cond/IdentityIdentity,model/synthesis/layer_2/igdn_2/cond:output:0*
T0
*
_output_shapes
: 2.
,model/synthesis/layer_2/igdn_2/cond/Identity?
%model/synthesis/layer_2/igdn_2/cond_1StatelessIf5model/synthesis/layer_2/igdn_2/cond/Identity:output:0(model/synthesis/layer_2/BiasAdd:output:0&model_synthesis_layer_2_igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_2_igdn_2_cond_1_false_200368*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_2_igdn_2_cond_1_true_2003672'
%model/synthesis/layer_2/igdn_2/cond_1?
.model/synthesis/layer_2/igdn_2/cond_1/IdentityIdentity.model/synthesis/layer_2/igdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200413*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200423*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
,model/synthesis/layer_2/igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,model/synthesis/layer_2/igdn_2/Reshape/shape?
&model/synthesis/layer_2/igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:05model/synthesis/layer_2/igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&model/synthesis/layer_2/igdn_2/Reshape?
*model/synthesis/layer_2/igdn_2/convolutionConv2D7model/synthesis/layer_2/igdn_2/cond_1/Identity:output:0/model/synthesis/layer_2/igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*model/synthesis/layer_2/igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200437*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
&model/synthesis/layer_2/igdn_2/BiasAddBiasAdd3model/synthesis/layer_2/igdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&model/synthesis/layer_2/igdn_2/BiasAdd?
"model/synthesis/layer_2/igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"model/synthesis/layer_2/igdn_2/x_1?
&model/synthesis/layer_2/igdn_2/Equal_1Equal(model_synthesis_layer_2_igdn_2_equal_1_x+model/synthesis/layer_2/igdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&model/synthesis/layer_2/igdn_2/Equal_1?
%model/synthesis/layer_2/igdn_2/cond_2StatelessIf*model/synthesis/layer_2/igdn_2/Equal_1:z:0/model/synthesis/layer_2/igdn_2/BiasAdd:output:0(model_synthesis_layer_2_igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *E
else_branch6R4
2model_synthesis_layer_2_igdn_2_cond_2_false_200451*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_2_igdn_2_cond_2_true_2004502'
%model/synthesis/layer_2/igdn_2/cond_2?
.model/synthesis/layer_2/igdn_2/cond_2/IdentityIdentity.model/synthesis/layer_2/igdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_2/Identity?
"model/synthesis/layer_2/igdn_2/mulMul(model/synthesis/layer_2/BiasAdd:output:07model/synthesis/layer_2/igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"model/synthesis/layer_2/igdn_2/mul?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshape?
&model/synthesis/layer_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&model/synthesis/layer_3/transpose/perm?
!model/synthesis/layer_3/transpose	Transposelayer_3/kernel/Reshape:output:0/model/synthesis/layer_3/transpose/perm:output:0*
T0*'
_output_shapes
:?2#
!model/synthesis/layer_3/transpose?
model/synthesis/layer_3/ShapeShape&model/synthesis/layer_2/igdn_2/mul:z:0*
T0*
_output_shapes
:2
model/synthesis/layer_3/Shape?
+model/synthesis/layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model/synthesis/layer_3/strided_slice/stack?
-model/synthesis/layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_3/strided_slice/stack_1?
-model/synthesis/layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_3/strided_slice/stack_2?
%model/synthesis/layer_3/strided_sliceStridedSlice&model/synthesis/layer_3/Shape:output:04model/synthesis/layer_3/strided_slice/stack:output:06model/synthesis/layer_3/strided_slice/stack_1:output:06model/synthesis/layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model/synthesis/layer_3/strided_slice?
-model/synthesis/layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_3/strided_slice_1/stack?
/model/synthesis/layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_3/strided_slice_1/stack_1?
/model/synthesis/layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_3/strided_slice_1/stack_2?
'model/synthesis/layer_3/strided_slice_1StridedSlice&model/synthesis/layer_3/Shape:output:06model/synthesis/layer_3/strided_slice_1/stack:output:08model/synthesis/layer_3/strided_slice_1/stack_1:output:08model/synthesis/layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_3/strided_slice_1?
model/synthesis/layer_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
model/synthesis/layer_3/mul/y?
model/synthesis/layer_3/mulMul0model/synthesis/layer_3/strided_slice_1:output:0&model/synthesis/layer_3/mul/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_3/mul?
model/synthesis/layer_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
model/synthesis/layer_3/add/y?
model/synthesis/layer_3/addAddV2model/synthesis/layer_3/mul:z:0&model/synthesis/layer_3/add/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_3/add?
-model/synthesis/layer_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model/synthesis/layer_3/strided_slice_2/stack?
/model/synthesis/layer_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_3/strided_slice_2/stack_1?
/model/synthesis/layer_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model/synthesis/layer_3/strided_slice_2/stack_2?
'model/synthesis/layer_3/strided_slice_2StridedSlice&model/synthesis/layer_3/Shape:output:06model/synthesis/layer_3/strided_slice_2/stack:output:08model/synthesis/layer_3/strided_slice_2/stack_1:output:08model/synthesis/layer_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model/synthesis/layer_3/strided_slice_2?
model/synthesis/layer_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2!
model/synthesis/layer_3/mul_1/y?
model/synthesis/layer_3/mul_1Mul0model/synthesis/layer_3/strided_slice_2:output:0(model/synthesis/layer_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_3/mul_1?
model/synthesis/layer_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2!
model/synthesis/layer_3/add_1/y?
model/synthesis/layer_3/add_1AddV2!model/synthesis/layer_3/mul_1:z:0(model/synthesis/layer_3/add_1/y:output:0*
T0*
_output_shapes
: 2
model/synthesis/layer_3/add_1?
6model/synthesis/layer_3/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :28
6model/synthesis/layer_3/conv2d_transpose/input_sizes/3?
4model/synthesis/layer_3/conv2d_transpose/input_sizesPack.model/synthesis/layer_3/strided_slice:output:0model/synthesis/layer_3/add:z:0!model/synthesis/layer_3/add_1:z:0?model/synthesis/layer_3/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:26
4model/synthesis/layer_3/conv2d_transpose/input_sizes?
(model/synthesis/layer_3/conv2d_transposeConv2DBackpropInput=model/synthesis/layer_3/conv2d_transpose/input_sizes:output:0%model/synthesis/layer_3/transpose:y:0&model/synthesis/layer_2/igdn_2/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2*
(model/synthesis/layer_3/conv2d_transpose?
.model/synthesis/layer_3/BiasAdd/ReadVariableOpReadVariableOp7model_synthesis_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model/synthesis/layer_3/BiasAdd/ReadVariableOp?
model/synthesis/layer_3/BiasAddBiasAdd1model/synthesis/layer_3/conv2d_transpose:output:06model/synthesis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2!
model/synthesis/layer_3/BiasAdd?
model/synthesis/lambda_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2 
model/synthesis/lambda_1/mul/y?
model/synthesis/lambda_1/mulMul(model/synthesis/layer_3/BiasAdd:output:0'model/synthesis/lambda_1/mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
model/synthesis/lambda_1/mul?
IdentityIdentity model/synthesis/lambda_1/mul:z:0/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp/^model/synthesis/layer_0/BiasAdd/ReadVariableOp/^model/synthesis/layer_1/BiasAdd/ReadVariableOp/^model/synthesis/layer_2/BiasAdd/ReadVariableOp/^model/synthesis/layer_3/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp2`
.model/synthesis/layer_0/BiasAdd/ReadVariableOp.model/synthesis/layer_0/BiasAdd/ReadVariableOp2`
.model/synthesis/layer_1/BiasAdd/ReadVariableOp.model/synthesis/layer_1/BiasAdd/ReadVariableOp2`
.model/synthesis/layer_2/BiasAdd/ReadVariableOp.model/synthesis/layer_2/BiasAdd/ReadVariableOp2`
.model/synthesis/layer_3/BiasAdd/ReadVariableOp.model/synthesis/layer_3/BiasAdd/ReadVariableOp:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
)synthesis_layer_0_igdn_0_cond_true_202132-
)synthesis_layer_0_igdn_0_cond_placeholder
*
&synthesis_layer_0_igdn_0_cond_identity
?
#synthesis/layer_0/igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2%
#synthesis/layer_0/igdn_0/cond/Const?
&synthesis/layer_0/igdn_0/cond/IdentityIdentity,synthesis/layer_0/igdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity"Y
&synthesis_layer_0_igdn_0_cond_identity/synthesis/layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_1_igdn_1_cond_2_true_202387M
Isynthesis_layer_1_igdn_1_cond_2_identity_synthesis_layer_1_igdn_1_biasadd/
+synthesis_layer_1_igdn_1_cond_2_placeholder,
(synthesis_layer_1_igdn_1_cond_2_identity?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentityIsynthesis_layer_1_igdn_1_cond_2_identity_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity"]
(synthesis_layer_1_igdn_1_cond_2_identity1synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_2_igdn_2_cond_1_true_2040372
.layer_2_igdn_2_cond_1_identity_layer_2_biasadd%
!layer_2_igdn_2_cond_1_placeholder"
layer_2_igdn_2_cond_1_identity?
layer_2/igdn_2/cond_1/IdentityIdentity.layer_2_igdn_2_cond_1_identity_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity"I
layer_2_igdn_2_cond_1_identity'layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
??
?
A__inference_model_layer_call_and_return_conditional_losses_202615

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_0_biasadd_readvariableop_resource:	?$
 synthesis_layer_0_igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y&
"synthesis_layer_0_igdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_1_biasadd_readvariableop_resource:	?$
 synthesis_layer_1_igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y&
"synthesis_layer_1_igdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??@
1synthesis_layer_2_biasadd_readvariableop_resource:	?$
 synthesis_layer_2_igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y&
"synthesis_layer_2_igdn_2_equal_1_x
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	??
1synthesis_layer_3_biasadd_readvariableop_resource:
identity??.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?(synthesis/layer_0/BiasAdd/ReadVariableOp?(synthesis/layer_1/BiasAdd/ReadVariableOp?(synthesis/layer_2/BiasAdd/ReadVariableOp?(synthesis/layer_3/BiasAdd/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshape?
 synthesis/layer_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_0/transpose/perm?
synthesis/layer_0/transpose	Transposelayer_0/kernel/Reshape:output:0)synthesis/layer_0/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_0/transposeh
synthesis/layer_0/ShapeShapeinputs*
T0*
_output_shapes
:2
synthesis/layer_0/Shape?
%synthesis/layer_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_0/strided_slice/stack?
'synthesis/layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice/stack_1?
'synthesis/layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice/stack_2?
synthesis/layer_0/strided_sliceStridedSlice synthesis/layer_0/Shape:output:0.synthesis/layer_0/strided_slice/stack:output:00synthesis/layer_0/strided_slice/stack_1:output:00synthesis/layer_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_0/strided_slice?
'synthesis/layer_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice_1/stack?
)synthesis/layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_1/stack_1?
)synthesis/layer_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_1/stack_2?
!synthesis/layer_0/strided_slice_1StridedSlice synthesis/layer_0/Shape:output:00synthesis/layer_0/strided_slice_1/stack:output:02synthesis/layer_0/strided_slice_1/stack_1:output:02synthesis/layer_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_0/strided_slice_1t
synthesis/layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_0/mul/y?
synthesis/layer_0/mulMul*synthesis/layer_0/strided_slice_1:output:0 synthesis/layer_0/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/mult
synthesis/layer_0/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_0/add/y?
synthesis/layer_0/addAddV2synthesis/layer_0/mul:z:0 synthesis/layer_0/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/add?
'synthesis/layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_0/strided_slice_2/stack?
)synthesis/layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_2/stack_1?
)synthesis/layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_0/strided_slice_2/stack_2?
!synthesis/layer_0/strided_slice_2StridedSlice synthesis/layer_0/Shape:output:00synthesis/layer_0/strided_slice_2/stack:output:02synthesis/layer_0/strided_slice_2/stack_1:output:02synthesis/layer_0/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_0/strided_slice_2x
synthesis/layer_0/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_0/mul_1/y?
synthesis/layer_0/mul_1Mul*synthesis/layer_0/strided_slice_2:output:0"synthesis/layer_0/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/mul_1x
synthesis/layer_0/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_0/add_1/y?
synthesis/layer_0/add_1AddV2synthesis/layer_0/mul_1:z:0"synthesis/layer_0/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_0/add_1?
0synthesis/layer_0/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_0/conv2d_transpose/input_sizes/3?
.synthesis/layer_0/conv2d_transpose/input_sizesPack(synthesis/layer_0/strided_slice:output:0synthesis/layer_0/add:z:0synthesis/layer_0/add_1:z:09synthesis/layer_0/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_0/conv2d_transpose/input_sizes?
"synthesis/layer_0/conv2d_transposeConv2DBackpropInput7synthesis/layer_0/conv2d_transpose/input_sizes:output:0synthesis/layer_0/transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_0/conv2d_transpose?
(synthesis/layer_0/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_0/BiasAdd/ReadVariableOp?
synthesis/layer_0/BiasAddBiasAdd+synthesis/layer_0/conv2d_transpose:output:00synthesis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_0/BiasAdd}
synthesis/layer_0/igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_0/igdn_0/x?
synthesis/layer_0/igdn_0/EqualEqual synthesis_layer_0_igdn_0_equal_x#synthesis/layer_0/igdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_0/igdn_0/Equal?
synthesis/layer_0/igdn_0/condStatelessIf"synthesis/layer_0/igdn_0/Equal:z:0"synthesis/layer_0/igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_0_igdn_0_cond_false_202133*
output_shapes
: *<
then_branch-R+
)synthesis_layer_0_igdn_0_cond_true_2021322
synthesis/layer_0/igdn_0/cond?
&synthesis/layer_0/igdn_0/cond/IdentityIdentity&synthesis/layer_0/igdn_0/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_0/igdn_0/cond/Identity?
synthesis/layer_0/igdn_0/cond_1StatelessIf/synthesis/layer_0/igdn_0/cond/Identity:output:0"synthesis/layer_0/BiasAdd:output:0 synthesis_layer_0_igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_0_igdn_0_cond_1_false_202144*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_1_true_2021432!
synthesis/layer_0/igdn_0/cond_1?
(synthesis/layer_0/igdn_0/cond_1/IdentityIdentity(synthesis/layer_0/igdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202189*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202199*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
&synthesis/layer_0/igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_0/igdn_0/Reshape/shape?
 synthesis/layer_0/igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0/synthesis/layer_0/igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_0/igdn_0/Reshape?
$synthesis/layer_0/igdn_0/convolutionConv2D1synthesis/layer_0/igdn_0/cond_1/Identity:output:0)synthesis/layer_0/igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_0/igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202213*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
 synthesis/layer_0/igdn_0/BiasAddBiasAdd-synthesis/layer_0/igdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_0/igdn_0/BiasAdd?
synthesis/layer_0/igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_0/igdn_0/x_1?
 synthesis/layer_0/igdn_0/Equal_1Equal"synthesis_layer_0_igdn_0_equal_1_x%synthesis/layer_0/igdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_0/igdn_0/Equal_1?
synthesis/layer_0/igdn_0/cond_2StatelessIf$synthesis/layer_0/igdn_0/Equal_1:z:0)synthesis/layer_0/igdn_0/BiasAdd:output:0"synthesis_layer_0_igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_0_igdn_0_cond_2_false_202227*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_2_true_2022262!
synthesis/layer_0/igdn_0/cond_2?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentity(synthesis/layer_0/igdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity?
synthesis/layer_0/igdn_0/mulMul"synthesis/layer_0/BiasAdd:output:01synthesis/layer_0/igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_0/igdn_0/mul?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshape?
 synthesis/layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_1/transpose/perm?
synthesis/layer_1/transpose	Transposelayer_1/kernel/Reshape:output:0)synthesis/layer_1/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_1/transpose?
synthesis/layer_1/ShapeShape synthesis/layer_0/igdn_0/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_1/Shape?
%synthesis/layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_1/strided_slice/stack?
'synthesis/layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice/stack_1?
'synthesis/layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice/stack_2?
synthesis/layer_1/strided_sliceStridedSlice synthesis/layer_1/Shape:output:0.synthesis/layer_1/strided_slice/stack:output:00synthesis/layer_1/strided_slice/stack_1:output:00synthesis/layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_1/strided_slice?
'synthesis/layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice_1/stack?
)synthesis/layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_1/stack_1?
)synthesis/layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_1/stack_2?
!synthesis/layer_1/strided_slice_1StridedSlice synthesis/layer_1/Shape:output:00synthesis/layer_1/strided_slice_1/stack:output:02synthesis/layer_1/strided_slice_1/stack_1:output:02synthesis/layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_1/strided_slice_1t
synthesis/layer_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_1/mul/y?
synthesis/layer_1/mulMul*synthesis/layer_1/strided_slice_1:output:0 synthesis/layer_1/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/mult
synthesis/layer_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_1/add/y?
synthesis/layer_1/addAddV2synthesis/layer_1/mul:z:0 synthesis/layer_1/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/add?
'synthesis/layer_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_1/strided_slice_2/stack?
)synthesis/layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_2/stack_1?
)synthesis/layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_1/strided_slice_2/stack_2?
!synthesis/layer_1/strided_slice_2StridedSlice synthesis/layer_1/Shape:output:00synthesis/layer_1/strided_slice_2/stack:output:02synthesis/layer_1/strided_slice_2/stack_1:output:02synthesis/layer_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_1/strided_slice_2x
synthesis/layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_1/mul_1/y?
synthesis/layer_1/mul_1Mul*synthesis/layer_1/strided_slice_2:output:0"synthesis/layer_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/mul_1x
synthesis/layer_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_1/add_1/y?
synthesis/layer_1/add_1AddV2synthesis/layer_1/mul_1:z:0"synthesis/layer_1/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_1/add_1?
0synthesis/layer_1/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_1/conv2d_transpose/input_sizes/3?
.synthesis/layer_1/conv2d_transpose/input_sizesPack(synthesis/layer_1/strided_slice:output:0synthesis/layer_1/add:z:0synthesis/layer_1/add_1:z:09synthesis/layer_1/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_1/conv2d_transpose/input_sizes?
"synthesis/layer_1/conv2d_transposeConv2DBackpropInput7synthesis/layer_1/conv2d_transpose/input_sizes:output:0synthesis/layer_1/transpose:y:0 synthesis/layer_0/igdn_0/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_1/conv2d_transpose?
(synthesis/layer_1/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_1/BiasAdd/ReadVariableOp?
synthesis/layer_1/BiasAddBiasAdd+synthesis/layer_1/conv2d_transpose:output:00synthesis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_1/BiasAdd}
synthesis/layer_1/igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_1/igdn_1/x?
synthesis/layer_1/igdn_1/EqualEqual synthesis_layer_1_igdn_1_equal_x#synthesis/layer_1/igdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_1/igdn_1/Equal?
synthesis/layer_1/igdn_1/condStatelessIf"synthesis/layer_1/igdn_1/Equal:z:0"synthesis/layer_1/igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_1_igdn_1_cond_false_202294*
output_shapes
: *<
then_branch-R+
)synthesis_layer_1_igdn_1_cond_true_2022932
synthesis/layer_1/igdn_1/cond?
&synthesis/layer_1/igdn_1/cond/IdentityIdentity&synthesis/layer_1/igdn_1/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity?
synthesis/layer_1/igdn_1/cond_1StatelessIf/synthesis/layer_1/igdn_1/cond/Identity:output:0"synthesis/layer_1/BiasAdd:output:0 synthesis_layer_1_igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_1_igdn_1_cond_1_false_202305*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_1_true_2023042!
synthesis/layer_1/igdn_1/cond_1?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentity(synthesis/layer_1/igdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202350*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202360*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
&synthesis/layer_1/igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_1/igdn_1/Reshape/shape?
 synthesis/layer_1/igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0/synthesis/layer_1/igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_1/igdn_1/Reshape?
$synthesis/layer_1/igdn_1/convolutionConv2D1synthesis/layer_1/igdn_1/cond_1/Identity:output:0)synthesis/layer_1/igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_1/igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202374*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
 synthesis/layer_1/igdn_1/BiasAddBiasAdd-synthesis/layer_1/igdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_1/igdn_1/BiasAdd?
synthesis/layer_1/igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_1/igdn_1/x_1?
 synthesis/layer_1/igdn_1/Equal_1Equal"synthesis_layer_1_igdn_1_equal_1_x%synthesis/layer_1/igdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_1/igdn_1/Equal_1?
synthesis/layer_1/igdn_1/cond_2StatelessIf$synthesis/layer_1/igdn_1/Equal_1:z:0)synthesis/layer_1/igdn_1/BiasAdd:output:0"synthesis_layer_1_igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_1_igdn_1_cond_2_false_202388*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_2_true_2023872!
synthesis/layer_1/igdn_1/cond_2?
(synthesis/layer_1/igdn_1/cond_2/IdentityIdentity(synthesis/layer_1/igdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_2/Identity?
synthesis/layer_1/igdn_1/mulMul"synthesis/layer_1/BiasAdd:output:01synthesis/layer_1/igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_1/igdn_1/mul?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshape?
 synthesis/layer_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_2/transpose/perm?
synthesis/layer_2/transpose	Transposelayer_2/kernel/Reshape:output:0)synthesis/layer_2/transpose/perm:output:0*
T0*(
_output_shapes
:??2
synthesis/layer_2/transpose?
synthesis/layer_2/ShapeShape synthesis/layer_1/igdn_1/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_2/Shape?
%synthesis/layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_2/strided_slice/stack?
'synthesis/layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice/stack_1?
'synthesis/layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice/stack_2?
synthesis/layer_2/strided_sliceStridedSlice synthesis/layer_2/Shape:output:0.synthesis/layer_2/strided_slice/stack:output:00synthesis/layer_2/strided_slice/stack_1:output:00synthesis/layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_2/strided_slice?
'synthesis/layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice_1/stack?
)synthesis/layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_1/stack_1?
)synthesis/layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_1/stack_2?
!synthesis/layer_2/strided_slice_1StridedSlice synthesis/layer_2/Shape:output:00synthesis/layer_2/strided_slice_1/stack:output:02synthesis/layer_2/strided_slice_1/stack_1:output:02synthesis/layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_2/strided_slice_1t
synthesis/layer_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_2/mul/y?
synthesis/layer_2/mulMul*synthesis/layer_2/strided_slice_1:output:0 synthesis/layer_2/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/mult
synthesis/layer_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_2/add/y?
synthesis/layer_2/addAddV2synthesis/layer_2/mul:z:0 synthesis/layer_2/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/add?
'synthesis/layer_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_2/strided_slice_2/stack?
)synthesis/layer_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_2/stack_1?
)synthesis/layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_2/strided_slice_2/stack_2?
!synthesis/layer_2/strided_slice_2StridedSlice synthesis/layer_2/Shape:output:00synthesis/layer_2/strided_slice_2/stack:output:02synthesis/layer_2/strided_slice_2/stack_1:output:02synthesis/layer_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_2/strided_slice_2x
synthesis/layer_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_2/mul_1/y?
synthesis/layer_2/mul_1Mul*synthesis/layer_2/strided_slice_2:output:0"synthesis/layer_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/mul_1x
synthesis/layer_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_2/add_1/y?
synthesis/layer_2/add_1AddV2synthesis/layer_2/mul_1:z:0"synthesis/layer_2/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_2/add_1?
0synthesis/layer_2/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?22
0synthesis/layer_2/conv2d_transpose/input_sizes/3?
.synthesis/layer_2/conv2d_transpose/input_sizesPack(synthesis/layer_2/strided_slice:output:0synthesis/layer_2/add:z:0synthesis/layer_2/add_1:z:09synthesis/layer_2/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_2/conv2d_transpose/input_sizes?
"synthesis/layer_2/conv2d_transposeConv2DBackpropInput7synthesis/layer_2/conv2d_transpose/input_sizes:output:0synthesis/layer_2/transpose:y:0 synthesis/layer_1/igdn_1/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_2/conv2d_transpose?
(synthesis/layer_2/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02*
(synthesis/layer_2/BiasAdd/ReadVariableOp?
synthesis/layer_2/BiasAddBiasAdd+synthesis/layer_2/conv2d_transpose:output:00synthesis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_2/BiasAdd}
synthesis/layer_2/igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_2/igdn_2/x?
synthesis/layer_2/igdn_2/EqualEqual synthesis_layer_2_igdn_2_equal_x#synthesis/layer_2/igdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
synthesis/layer_2/igdn_2/Equal?
synthesis/layer_2/igdn_2/condStatelessIf"synthesis/layer_2/igdn_2/Equal:z:0"synthesis/layer_2/igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *=
else_branch.R,
*synthesis_layer_2_igdn_2_cond_false_202455*
output_shapes
: *<
then_branch-R+
)synthesis_layer_2_igdn_2_cond_true_2024542
synthesis/layer_2/igdn_2/cond?
&synthesis/layer_2/igdn_2/cond/IdentityIdentity&synthesis/layer_2/igdn_2/cond:output:0*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity?
synthesis/layer_2/igdn_2/cond_1StatelessIf/synthesis/layer_2/igdn_2/cond/Identity:output:0"synthesis/layer_2/BiasAdd:output:0 synthesis_layer_2_igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_2_igdn_2_cond_1_false_202466*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_1_true_2024652!
synthesis/layer_2/igdn_2/cond_1?
(synthesis/layer_2/igdn_2/cond_1/IdentityIdentity(synthesis/layer_2/igdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202511*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202521*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
&synthesis/layer_2/igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2(
&synthesis/layer_2/igdn_2/Reshape/shape?
 synthesis/layer_2/igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0/synthesis/layer_2/igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2"
 synthesis/layer_2/igdn_2/Reshape?
$synthesis/layer_2/igdn_2/convolutionConv2D1synthesis/layer_2/igdn_2/cond_1/Identity:output:0)synthesis/layer_2/igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2&
$synthesis/layer_2/igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-202535*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
 synthesis/layer_2/igdn_2/BiasAddBiasAdd-synthesis/layer_2/igdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 synthesis/layer_2/igdn_2/BiasAdd?
synthesis/layer_2/igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
synthesis/layer_2/igdn_2/x_1?
 synthesis/layer_2/igdn_2/Equal_1Equal"synthesis_layer_2_igdn_2_equal_1_x%synthesis/layer_2/igdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2"
 synthesis/layer_2/igdn_2/Equal_1?
synthesis/layer_2/igdn_2/cond_2StatelessIf$synthesis/layer_2/igdn_2/Equal_1:z:0)synthesis/layer_2/igdn_2/BiasAdd:output:0"synthesis_layer_2_igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *?
else_branch0R.
,synthesis_layer_2_igdn_2_cond_2_false_202549*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_2_true_2025482!
synthesis/layer_2/igdn_2/cond_2?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentity(synthesis/layer_2/igdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity?
synthesis/layer_2/igdn_2/mulMul"synthesis/layer_2/BiasAdd:output:01synthesis/layer_2/igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
synthesis/layer_2/igdn_2/mul?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshape?
 synthesis/layer_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 synthesis/layer_3/transpose/perm?
synthesis/layer_3/transpose	Transposelayer_3/kernel/Reshape:output:0)synthesis/layer_3/transpose/perm:output:0*
T0*'
_output_shapes
:?2
synthesis/layer_3/transpose?
synthesis/layer_3/ShapeShape synthesis/layer_2/igdn_2/mul:z:0*
T0*
_output_shapes
:2
synthesis/layer_3/Shape?
%synthesis/layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%synthesis/layer_3/strided_slice/stack?
'synthesis/layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice/stack_1?
'synthesis/layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice/stack_2?
synthesis/layer_3/strided_sliceStridedSlice synthesis/layer_3/Shape:output:0.synthesis/layer_3/strided_slice/stack:output:00synthesis/layer_3/strided_slice/stack_1:output:00synthesis/layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
synthesis/layer_3/strided_slice?
'synthesis/layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice_1/stack?
)synthesis/layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_1/stack_1?
)synthesis/layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_1/stack_2?
!synthesis/layer_3/strided_slice_1StridedSlice synthesis/layer_3/Shape:output:00synthesis/layer_3/strided_slice_1/stack:output:02synthesis/layer_3/strided_slice_1/stack_1:output:02synthesis/layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_3/strided_slice_1t
synthesis/layer_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_3/mul/y?
synthesis/layer_3/mulMul*synthesis/layer_3/strided_slice_1:output:0 synthesis/layer_3/mul/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/mult
synthesis/layer_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_3/add/y?
synthesis/layer_3/addAddV2synthesis/layer_3/mul:z:0 synthesis/layer_3/add/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/add?
'synthesis/layer_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'synthesis/layer_3/strided_slice_2/stack?
)synthesis/layer_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_2/stack_1?
)synthesis/layer_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)synthesis/layer_3/strided_slice_2/stack_2?
!synthesis/layer_3/strided_slice_2StridedSlice synthesis/layer_3/Shape:output:00synthesis/layer_3/strided_slice_2/stack:output:02synthesis/layer_3/strided_slice_2/stack_1:output:02synthesis/layer_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!synthesis/layer_3/strided_slice_2x
synthesis/layer_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
synthesis/layer_3/mul_1/y?
synthesis/layer_3/mul_1Mul*synthesis/layer_3/strided_slice_2:output:0"synthesis/layer_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/mul_1x
synthesis/layer_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
synthesis/layer_3/add_1/y?
synthesis/layer_3/add_1AddV2synthesis/layer_3/mul_1:z:0"synthesis/layer_3/add_1/y:output:0*
T0*
_output_shapes
: 2
synthesis/layer_3/add_1?
0synthesis/layer_3/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :22
0synthesis/layer_3/conv2d_transpose/input_sizes/3?
.synthesis/layer_3/conv2d_transpose/input_sizesPack(synthesis/layer_3/strided_slice:output:0synthesis/layer_3/add:z:0synthesis/layer_3/add_1:z:09synthesis/layer_3/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:20
.synthesis/layer_3/conv2d_transpose/input_sizes?
"synthesis/layer_3/conv2d_transposeConv2DBackpropInput7synthesis/layer_3/conv2d_transpose/input_sizes:output:0synthesis/layer_3/transpose:y:0 synthesis/layer_2/igdn_2/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2$
"synthesis/layer_3/conv2d_transpose?
(synthesis/layer_3/BiasAdd/ReadVariableOpReadVariableOp1synthesis_layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(synthesis/layer_3/BiasAdd/ReadVariableOp?
synthesis/layer_3/BiasAddBiasAdd+synthesis/layer_3/conv2d_transpose:output:00synthesis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2
synthesis/layer_3/BiasAddy
synthesis/lambda_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
synthesis/lambda_1/mul/y?
synthesis/lambda_1/mulMul"synthesis/layer_3/BiasAdd:output:0!synthesis/lambda_1/mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
synthesis/lambda_1/mul?
IdentityIdentitysynthesis/lambda_1/mul:z:0/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp)^synthesis/layer_0/BiasAdd/ReadVariableOp)^synthesis/layer_1/BiasAdd/ReadVariableOp)^synthesis/layer_2/BiasAdd/ReadVariableOp)^synthesis/layer_3/BiasAdd/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp2T
(synthesis/layer_0/BiasAdd/ReadVariableOp(synthesis/layer_0/BiasAdd/ReadVariableOp2T
(synthesis/layer_1/BiasAdd/ReadVariableOp(synthesis/layer_1/BiasAdd/ReadVariableOp2T
(synthesis/layer_2/BiasAdd/ReadVariableOp(synthesis/layer_2/BiasAdd/ReadVariableOp2T
(synthesis/layer_3/BiasAdd/ReadVariableOp(synthesis/layer_3/BiasAdd/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202687K
Gsynthesis_layer_0_igdn_0_cond_1_cond_cond_pow_synthesis_layer_0_biasadd3
/synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y6
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
-synthesis/layer_0/igdn_0/cond_1/cond/cond/powPowGsynthesis_layer_0_igdn_0_cond_1_cond_cond_pow_synthesis_layer_0_biasadd/synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/cond/pow?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity1synthesis/layer_0/igdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"q
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
'layer_0_igdn_0_cond_2_cond_false_2038089
5layer_0_igdn_0_cond_2_cond_pow_layer_0_igdn_0_biasadd$
 layer_0_igdn_0_cond_2_cond_pow_y'
#layer_0_igdn_0_cond_2_cond_identity?
layer_0/igdn_0/cond_2/cond/powPow5layer_0_igdn_0_cond_2_cond_pow_layer_0_igdn_0_biasadd layer_0_igdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/cond/pow?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity"layer_0/igdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity"S
#layer_0_igdn_0_cond_2_cond_identity,layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?i
?
C__inference_layer_0_layer_call_and_return_conditional_losses_200686

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y
igdn_0_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_0/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_0/x?
igdn_0/EqualEqualigdn_0_equal_xigdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/Equal?
igdn_0/condStatelessIfigdn_0/Equal:z:0igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_0_cond_false_200563*
output_shapes
: **
then_branchR
igdn_0_cond_true_2005622
igdn_0/condo
igdn_0/cond/IdentityIdentityigdn_0/cond:output:0*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity?
igdn_0/cond_1StatelessIfigdn_0/cond/Identity:output:0BiasAdd:output:0igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_0_cond_1_false_200574*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_1_true_2005732
igdn_0/cond_1?
igdn_0/cond_1/IdentityIdentityigdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200619*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200629*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_0/Reshape/shape?
igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_0/Reshape?
igdn_0/convolutionConv2Digdn_0/cond_1/Identity:output:0igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200643*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
igdn_0/BiasAddBiasAddigdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/BiasAdd]

igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_0/x_1?
igdn_0/Equal_1Equaligdn_0_equal_1_xigdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/Equal_1?
igdn_0/cond_2StatelessIfigdn_0/Equal_1:z:0igdn_0/BiasAdd:output:0igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_0_cond_2_false_200657*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_2_true_2006562
igdn_0/cond_2?
igdn_0/cond_2/IdentityIdentityigdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/Identity?

igdn_0/mulMulBiasAdd:output:0igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_0/mul?
IdentityIdentityigdn_0/mul:z:0^BiasAdd/ReadVariableOp/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
,synthesis_layer_0_igdn_0_cond_2_false_202227I
Esynthesis_layer_0_igdn_0_cond_2_cond_synthesis_layer_0_igdn_0_biasadd+
'synthesis_layer_0_igdn_0_cond_2_equal_x,
(synthesis_layer_0_igdn_0_cond_2_identity?
!synthesis/layer_0/igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!synthesis/layer_0/igdn_0/cond_2/x?
%synthesis/layer_0/igdn_0/cond_2/EqualEqual'synthesis_layer_0_igdn_0_cond_2_equal_x*synthesis/layer_0/igdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_0/igdn_0/cond_2/Equal?
$synthesis/layer_0/igdn_0/cond_2/condStatelessIf)synthesis/layer_0/igdn_0/cond_2/Equal:z:0Esynthesis_layer_0_igdn_0_cond_2_cond_synthesis_layer_0_igdn_0_biasadd'synthesis_layer_0_igdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_0_igdn_0_cond_2_cond_false_202236*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_2_cond_true_2022352&
$synthesis/layer_0/igdn_0/cond_2/cond?
-synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity-synthesis/layer_0/igdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_2/cond/Identity?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentity6synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity"]
(synthesis_layer_0_igdn_0_cond_2_identity1synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
1model_synthesis_layer_1_igdn_1_cond_2_true_200289Y
Umodel_synthesis_layer_1_igdn_1_cond_2_identity_model_synthesis_layer_1_igdn_1_biasadd5
1model_synthesis_layer_1_igdn_1_cond_2_placeholder2
.model_synthesis_layer_1_igdn_1_cond_2_identity?
.model/synthesis/layer_1/igdn_1/cond_2/IdentityIdentityUmodel_synthesis_layer_1_igdn_1_cond_2_identity_model_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_1/igdn_1/cond_2/Identity"i
.model_synthesis_layer_1_igdn_1_cond_2_identity7model/synthesis/layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_1_cond_false_200772#
igdn_1_cond_1_cond_cond_biasadd
igdn_1_cond_1_cond_equal_x
igdn_1_cond_1_cond_identityq
igdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_1/cond_1/cond/x?
igdn_1/cond_1/cond/EqualEqualigdn_1_cond_1_cond_equal_xigdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_1/cond/Equal?
igdn_1/cond_1/cond/condStatelessIfigdn_1/cond_1/cond/Equal:z:0igdn_1_cond_1_cond_cond_biasaddigdn_1_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_1_cond_1_cond_cond_false_200782*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_1_cond_1_cond_cond_true_2007812
igdn_1/cond_1/cond/cond?
 igdn_1/cond_1/cond/cond/IdentityIdentity igdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity?
igdn_1/cond_1/cond/IdentityIdentity)igdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity"C
igdn_1_cond_1_cond_identity$igdn_1/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_2_cond_2_cond_true_201043*
&igdn_2_cond_2_cond_sqrt_igdn_2_biasadd"
igdn_2_cond_2_cond_placeholder
igdn_2_cond_2_cond_identity?
igdn_2/cond_2/cond/SqrtSqrt&igdn_2_cond_2_cond_sqrt_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Sqrt?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity"C
igdn_2_cond_2_cond_identity$igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_2_cond_1_cond_true_200960"
igdn_2_cond_1_cond_abs_biasadd"
igdn_2_cond_1_cond_placeholder
igdn_2_cond_1_cond_identity?
igdn_2/cond_1/cond/AbsAbsigdn_2_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Abs?
igdn_2/cond_1/cond/IdentityIdentityigdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity"C
igdn_2_cond_1_cond_identity$igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_1_cond_1_cond_cond_true_204431*
&igdn_1_cond_1_cond_cond_square_biasadd'
#igdn_1_cond_1_cond_cond_placeholder$
 igdn_1_cond_1_cond_cond_identity?
igdn_1/cond_1/cond/cond/SquareSquare&igdn_1_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_1/cond_1/cond/cond/Square?
 igdn_1/cond_1/cond/cond/IdentityIdentity"igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity"M
 igdn_1_cond_1_cond_cond_identity)igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
*synthesis_layer_2_igdn_2_cond_false_202455I
Esynthesis_layer_2_igdn_2_cond_identity_synthesis_layer_2_igdn_2_equal
*
&synthesis_layer_2_igdn_2_cond_identity
?
&synthesis/layer_2/igdn_2/cond/IdentityIdentityEsynthesis_layer_2_igdn_2_cond_identity_synthesis_layer_2_igdn_2_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_2/igdn_2/cond/Identity"Y
&synthesis_layer_2_igdn_2_cond_identity/synthesis/layer_2/igdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202163K
Gsynthesis_layer_0_igdn_0_cond_1_cond_cond_pow_synthesis_layer_0_biasadd3
/synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y6
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity?
-synthesis/layer_0/igdn_0/cond_1/cond/cond/powPowGsynthesis_layer_0_igdn_0_cond_1_cond_cond_pow_synthesis_layer_0_biasadd/synthesis_layer_0_igdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_0/igdn_0/cond_1/cond/cond/pow?
2synthesis/layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity1synthesis/layer_0/igdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????24
2synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity"q
2synthesis_layer_0_igdn_0_cond_1_cond_cond_identity;synthesis/layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+synthesis_layer_2_igdn_2_cond_2_true_202548M
Isynthesis_layer_2_igdn_2_cond_2_identity_synthesis_layer_2_igdn_2_biasadd/
+synthesis_layer_2_igdn_2_cond_2_placeholder,
(synthesis_layer_2_igdn_2_cond_2_identity?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentityIsynthesis_layer_2_igdn_2_cond_2_identity_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity"]
(synthesis_layer_2_igdn_2_cond_2_identity1synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,layer_2_igdn_2_cond_1_cond_cond_false_2035337
3layer_2_igdn_2_cond_1_cond_cond_pow_layer_2_biasadd)
%layer_2_igdn_2_cond_1_cond_cond_pow_y,
(layer_2_igdn_2_cond_1_cond_cond_identity?
#layer_2/igdn_2/cond_1/cond/cond/powPow3layer_2_igdn_2_cond_1_cond_cond_pow_layer_2_biasadd%layer_2_igdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/cond/pow?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity'layer_2/igdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity"]
(layer_2_igdn_2_cond_1_cond_cond_identity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
*__inference_synthesis_layer_call_fn_201390
layer_0_input
unknown
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3:
??
	unknown_4
	unknown_5
	unknown_6:	?
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:
??

unknown_12:	?

unknown_13

unknown_14:
??

unknown_15

unknown_16

unknown_17:	?

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22:
??

unknown_23:	?

unknown_24

unknown_25:
??

unknown_26

unknown_27

unknown_28:	?

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33:	?

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2013152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:q m
B
_output_shapes0
.:,????????????????????????????
'
_user_specified_namelayer_0_input:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201144
layer_0_input
layer_0_200687"
layer_0_200689:
??
layer_0_200691:	?
layer_0_200693"
layer_0_200695:
??
layer_0_200697
layer_0_200699
layer_0_200701:	?
layer_0_200703
layer_0_200705
layer_0_200707
layer_1_200876"
layer_1_200878:
??
layer_1_200880:	?
layer_1_200882"
layer_1_200884:
??
layer_1_200886
layer_1_200888
layer_1_200890:	?
layer_1_200892
layer_1_200894
layer_1_200896
layer_2_201065"
layer_2_201067:
??
layer_2_201069:	?
layer_2_201071"
layer_2_201073:
??
layer_2_201075
layer_2_201077
layer_2_201079:	?
layer_2_201081
layer_2_201083
layer_2_201085
layer_3_201128!
layer_3_201130:	?
layer_3_201132:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputlayer_0_200687layer_0_200689layer_0_200691layer_0_200693layer_0_200695layer_0_200697layer_0_200699layer_0_200701layer_0_200703layer_0_200705layer_0_200707*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_0_layer_call_and_return_conditional_losses_2006862!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_200876layer_1_200878layer_1_200880layer_1_200882layer_1_200884layer_1_200886layer_1_200888layer_1_200890layer_1_200892layer_1_200894layer_1_200896*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_1_layer_call_and_return_conditional_losses_2008752!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201065layer_2_201067layer_2_201069layer_2_201071layer_2_201073layer_2_201075layer_2_201077layer_2_201079layer_2_201081layer_2_201083layer_2_201085*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_2_layer_call_and_return_conditional_losses_2010642!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201128layer_3_201130layer_3_201132*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_2011272!
layer_3/StatefulPartitionedCall?
lambda_1/PartitionedCallPartitionedCall(layer_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011412
lambda_1/PartitionedCall?
IdentityIdentity!lambda_1/PartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:q m
B
_output_shapes0
.:,????????????????????????????
'
_user_specified_namelayer_0_input:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
igdn_0_cond_2_cond_true_200665*
&igdn_0_cond_2_cond_sqrt_igdn_0_biasadd"
igdn_0_cond_2_cond_placeholder
igdn_0_cond_2_cond_identity?
igdn_0/cond_2/cond/SqrtSqrt&igdn_0_cond_2_cond_sqrt_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Sqrt?
igdn_0/cond_2/cond/IdentityIdentityigdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_2/cond/Identity"C
igdn_0_cond_2_cond_identity$igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
$__inference_signature_wrapper_202091
input_2
unknown
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3:
??
	unknown_4
	unknown_5
	unknown_6:	?
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11:
??

unknown_12:	?

unknown_13

unknown_14:
??

unknown_15

unknown_16

unknown_17:	?

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22:
??

unknown_23:	?

unknown_24

unknown_25:
??

unknown_26

unknown_27

unknown_28:	?

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33:	?

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_2005172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
,synthesis_layer_1_igdn_1_cond_1_false_202829B
>synthesis_layer_1_igdn_1_cond_1_cond_synthesis_layer_1_biasadd+
'synthesis_layer_1_igdn_1_cond_1_equal_x,
(synthesis_layer_1_igdn_1_cond_1_identity?
!synthesis/layer_1/igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!synthesis/layer_1/igdn_1/cond_1/x?
%synthesis/layer_1/igdn_1/cond_1/EqualEqual'synthesis_layer_1_igdn_1_cond_1_equal_x*synthesis/layer_1/igdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2'
%synthesis/layer_1/igdn_1/cond_1/Equal?
$synthesis/layer_1/igdn_1/cond_1/condStatelessIf)synthesis/layer_1/igdn_1/cond_1/Equal:z:0>synthesis_layer_1_igdn_1_cond_1_cond_synthesis_layer_1_biasadd'synthesis_layer_1_igdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *D
else_branch5R3
1synthesis_layer_1_igdn_1_cond_1_cond_false_202838*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_1_cond_true_2028372&
$synthesis/layer_1/igdn_1/cond_1/cond?
-synthesis/layer_1/igdn_1/cond_1/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_1/cond/Identity?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentity6synthesis/layer_1/igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity"]
(synthesis_layer_1_igdn_1_cond_1_identity1synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_2_igdn_2_cond_1_false_204038.
*layer_2_igdn_2_cond_1_cond_layer_2_biasadd!
layer_2_igdn_2_cond_1_equal_x"
layer_2_igdn_2_cond_1_identityw
layer_2/igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/cond_1/x?
layer_2/igdn_2/cond_1/EqualEquallayer_2_igdn_2_cond_1_equal_x layer_2/igdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/cond_1/Equal?
layer_2/igdn_2/cond_1/condStatelessIflayer_2/igdn_2/cond_1/Equal:z:0*layer_2_igdn_2_cond_1_cond_layer_2_biasaddlayer_2_igdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_2_igdn_2_cond_1_cond_false_204047*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_1_cond_true_2040462
layer_2/igdn_2/cond_1/cond?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity#layer_2/igdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity?
layer_2/igdn_2/cond_1/IdentityIdentity,layer_2/igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity"I
layer_2_igdn_2_cond_1_identity'layer_2/igdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?i
?
C__inference_layer_1_layer_call_and_return_conditional_losses_200875

inputs
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y
igdn_1_equal_1_x
identity??BiasAdd/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshapey
transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
transpose/perm?
	transpose	Transposelayer_1/kernel/Reshape:output:0transpose/perm:output:0*
T0*(
_output_shapes
:??2
	transposeD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2T
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
conv2d_transpose/input_sizes/3?
conv2d_transpose/input_sizesPackstrided_slice:output:0add:z:0	add_1:z:0'conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/input_sizes?
conv2d_transposeConv2DBackpropInput%conv2d_transpose/input_sizes:output:0transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddY
igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_1/x?
igdn_1/EqualEqualigdn_1_equal_xigdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/Equal?
igdn_1/condStatelessIfigdn_1/Equal:z:0igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *+
else_branchR
igdn_1_cond_false_200752*
output_shapes
: **
then_branchR
igdn_1_cond_true_2007512
igdn_1/condo
igdn_1/cond/IdentityIdentityigdn_1/cond:output:0*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity?
igdn_1/cond_1StatelessIfigdn_1/cond/Identity:output:0BiasAdd:output:0igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_1_cond_1_false_200763*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_1_true_2007622
igdn_1/cond_1?
igdn_1/cond_1/IdentityIdentityigdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200808*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200818*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
igdn_1/Reshape/shape?
igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
igdn_1/Reshape?
igdn_1/convolutionConv2Digdn_1/cond_1/Identity:output:0igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-200832*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
igdn_1/BiasAddBiasAddigdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/BiasAdd]

igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2

igdn_1/x_1?
igdn_1/Equal_1Equaligdn_1_equal_1_xigdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/Equal_1?
igdn_1/cond_2StatelessIfigdn_1/Equal_1:z:0igdn_1/BiasAdd:output:0igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *-
else_branchR
igdn_1_cond_2_false_200846*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_2_true_2008452
igdn_1/cond_2?
igdn_1/cond_2/IdentityIdentityigdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/Identity?

igdn_1/mulMulBiasAdd:output:0igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2

igdn_1/mul?
IdentityIdentityigdn_1/mul:z:0^BiasAdd/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: 
?
?
igdn_2_cond_2_cond_true_204673*
&igdn_2_cond_2_cond_sqrt_igdn_2_biasadd"
igdn_2_cond_2_cond_placeholder
igdn_2_cond_2_cond_identity?
igdn_2/cond_2/cond/SqrtSqrt&igdn_2_cond_2_cond_sqrt_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Sqrt?
igdn_2/cond_2/cond/IdentityIdentityigdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_2/cond/Identity"C
igdn_2_cond_2_cond_identity$igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_1_igdn_1_cond_2_false_2039605
1layer_1_igdn_1_cond_2_cond_layer_1_igdn_1_biasadd!
layer_1_igdn_1_cond_2_equal_x"
layer_1_igdn_1_cond_2_identityw
layer_1/igdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_1/igdn_1/cond_2/x?
layer_1/igdn_1/cond_2/EqualEquallayer_1_igdn_1_cond_2_equal_x layer_1/igdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/cond_2/Equal?
layer_1/igdn_1/cond_2/condStatelessIflayer_1/igdn_1/cond_2/Equal:z:01layer_1_igdn_1_cond_2_cond_layer_1_igdn_1_biasaddlayer_1_igdn_1_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_1_igdn_1_cond_2_cond_false_203969*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_2_cond_true_2039682
layer_1/igdn_1/cond_2/cond?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity#layer_1/igdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity?
layer_1/igdn_1/cond_2/IdentityIdentity,layer_1/igdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity"I
layer_1_igdn_1_cond_2_identity'layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_2_igdn_2_cond_2_cond_true_203605:
6layer_2_igdn_2_cond_2_cond_sqrt_layer_2_igdn_2_biasadd*
&layer_2_igdn_2_cond_2_cond_placeholder'
#layer_2_igdn_2_cond_2_cond_identity?
layer_2/igdn_2/cond_2/cond/SqrtSqrt6layer_2_igdn_2_cond_2_cond_sqrt_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_2/igdn_2/cond_2/cond/Sqrt?
#layer_2/igdn_2/cond_2/cond/IdentityIdentity#layer_2/igdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_2/cond/Identity"S
#layer_2_igdn_2_cond_2_cond_identity,layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_1_cond_false_204402%
!igdn_1_cond_identity_igdn_1_equal

igdn_1_cond_identity
|
igdn_1/cond/IdentityIdentity!igdn_1_cond_identity_igdn_1_equal*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity"5
igdn_1_cond_identityigdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
!layer_0_igdn_0_cond_2_true_2037989
5layer_0_igdn_0_cond_2_identity_layer_0_igdn_0_biasadd%
!layer_0_igdn_0_cond_2_placeholder"
layer_0_igdn_0_cond_2_identity?
layer_0/igdn_0/cond_2/IdentityIdentity5layer_0_igdn_0_cond_2_identity_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity"I
layer_0_igdn_0_cond_2_identity'layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_0_cond_1_false_200574
igdn_0_cond_1_cond_biasadd
igdn_0_cond_1_equal_x
igdn_0_cond_1_identityg
igdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_0/cond_1/x?
igdn_0/cond_1/EqualEqualigdn_0_cond_1_equal_xigdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_1/Equal?
igdn_0/cond_1/condStatelessIfigdn_0/cond_1/Equal:z:0igdn_0_cond_1_cond_biasaddigdn_0_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_0_cond_1_cond_false_200583*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_1_cond_true_2005822
igdn_0/cond_1/cond?
igdn_0/cond_1/cond/IdentityIdentityigdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity?
igdn_0/cond_1/IdentityIdentity$igdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity"9
igdn_0_cond_1_identityigdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+layer_0_igdn_0_cond_1_cond_cond_true_203210:
6layer_0_igdn_0_cond_1_cond_cond_square_layer_0_biasadd/
+layer_0_igdn_0_cond_1_cond_cond_placeholder,
(layer_0_igdn_0_cond_1_cond_cond_identity?
&layer_0/igdn_0/cond_1/cond/cond/SquareSquare6layer_0_igdn_0_cond_1_cond_cond_square_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_0/igdn_0/cond_1/cond/cond/Square?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity*layer_0/igdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity"]
(layer_0_igdn_0_cond_1_cond_cond_identity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6model_synthesis_layer_0_igdn_0_cond_2_cond_true_200137Z
Vmodel_synthesis_layer_0_igdn_0_cond_2_cond_sqrt_model_synthesis_layer_0_igdn_0_biasadd:
6model_synthesis_layer_0_igdn_0_cond_2_cond_placeholder7
3model_synthesis_layer_0_igdn_0_cond_2_cond_identity?
/model/synthesis/layer_0/igdn_0/cond_2/cond/SqrtSqrtVmodel_synthesis_layer_0_igdn_0_cond_2_cond_sqrt_model_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/model/synthesis/layer_0/igdn_0/cond_2/cond/Sqrt?
3model/synthesis/layer_0/igdn_0/cond_2/cond/IdentityIdentity3model/synthesis/layer_0/igdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_0/igdn_0/cond_2/cond/Identity"s
3model_synthesis_layer_0_igdn_0_cond_2_cond_identity<model/synthesis/layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+layer_2_igdn_2_cond_1_cond_cond_true_203532:
6layer_2_igdn_2_cond_1_cond_cond_square_layer_2_biasadd/
+layer_2_igdn_2_cond_1_cond_cond_placeholder,
(layer_2_igdn_2_cond_1_cond_cond_identity?
&layer_2/igdn_2/cond_1/cond/cond/SquareSquare6layer_2_igdn_2_cond_1_cond_cond_square_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&layer_2/igdn_2/cond_1/cond/cond/Square?
(layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity*layer_2/igdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_2/igdn_2/cond_1/cond/cond/Identity"]
(layer_2_igdn_2_cond_1_cond_cond_identity1layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?	
?
7model_synthesis_layer_2_igdn_2_cond_2_cond_false_200460Y
Umodel_synthesis_layer_2_igdn_2_cond_2_cond_pow_model_synthesis_layer_2_igdn_2_biasadd4
0model_synthesis_layer_2_igdn_2_cond_2_cond_pow_y7
3model_synthesis_layer_2_igdn_2_cond_2_cond_identity?
.model/synthesis/layer_2/igdn_2/cond_2/cond/powPowUmodel_synthesis_layer_2_igdn_2_cond_2_cond_pow_model_synthesis_layer_2_igdn_2_biasadd0model_synthesis_layer_2_igdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_2/cond/pow?
3model/synthesis/layer_2/igdn_2/cond_2/cond/IdentityIdentity2model/synthesis/layer_2/igdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_2/cond/Identity"s
3model_synthesis_layer_2_igdn_2_cond_2_cond_identity<model/synthesis/layer_2/igdn_2/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
#igdn_1_cond_1_cond_cond_true_200781*
&igdn_1_cond_1_cond_cond_square_biasadd'
#igdn_1_cond_1_cond_cond_placeholder$
 igdn_1_cond_1_cond_cond_identity?
igdn_1/cond_1/cond/cond/SquareSquare&igdn_1_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
igdn_1/cond_1/cond/cond/Square?
 igdn_1/cond_1/cond/cond/IdentityIdentity"igdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_1/cond_1/cond/cond/Identity"M
 igdn_1_cond_1_cond_cond_identity)igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_0_igdn_0_cond_2_cond_true_203283:
6layer_0_igdn_0_cond_2_cond_sqrt_layer_0_igdn_0_biasadd*
&layer_0_igdn_0_cond_2_cond_placeholder'
#layer_0_igdn_0_cond_2_cond_identity?
layer_0/igdn_0/cond_2/cond/SqrtSqrt6layer_0_igdn_0_cond_2_cond_sqrt_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_0/igdn_0/cond_2/cond/Sqrt?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity#layer_0/igdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity"S
#layer_0_igdn_0_cond_2_cond_identity,layer_0/igdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_1_cond_1_false_204413
igdn_1_cond_1_cond_biasadd
igdn_1_cond_1_equal_x
igdn_1_cond_1_identityg
igdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_1/cond_1/x?
igdn_1/cond_1/EqualEqualigdn_1_cond_1_equal_xigdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_1/cond_1/Equal?
igdn_1/cond_1/condStatelessIfigdn_1/cond_1/Equal:z:0igdn_1_cond_1_cond_biasaddigdn_1_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_1_cond_1_cond_false_204422*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_1_cond_true_2044212
igdn_1/cond_1/cond?
igdn_1/cond_1/cond/IdentityIdentityigdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/cond/Identity?
igdn_1/cond_1/IdentityIdentity$igdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_1/Identity"9
igdn_1_cond_1_identityigdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
[
igdn_2_cond_false_204571%
!igdn_2_cond_identity_igdn_2_equal

igdn_2_cond_identity
|
igdn_2/cond/IdentityIdentity!igdn_2_cond_identity_igdn_2_equal*
T0
*
_output_shapes
: 2
igdn_2/cond/Identity"5
igdn_2_cond_identityigdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_1_igdn_1_cond_1_true_202304F
Bsynthesis_layer_1_igdn_1_cond_1_identity_synthesis_layer_1_biasadd/
+synthesis_layer_1_igdn_1_cond_1_placeholder,
(synthesis_layer_1_igdn_1_cond_1_identity?
(synthesis/layer_1/igdn_1/cond_1/IdentityIdentityBsynthesis_layer_1_igdn_1_cond_1_identity_synthesis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_1/igdn_1/cond_1/Identity"]
(synthesis_layer_1_igdn_1_cond_1_identity1synthesis/layer_1/igdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
7model_synthesis_layer_2_igdn_2_cond_1_cond_false_200377S
Omodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_model_synthesis_layer_2_biasadd6
2model_synthesis_layer_2_igdn_2_cond_1_cond_equal_x7
3model_synthesis_layer_2_igdn_2_cond_1_cond_identity?
,model/synthesis/layer_2/igdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,model/synthesis/layer_2/igdn_2/cond_1/cond/x?
0model/synthesis/layer_2/igdn_2/cond_1/cond/EqualEqual2model_synthesis_layer_2_igdn_2_cond_1_cond_equal_x5model/synthesis/layer_2/igdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0model/synthesis/layer_2/igdn_2/cond_1/cond/Equal?
/model/synthesis/layer_2/igdn_2/cond_1/cond/condStatelessIf4model/synthesis/layer_2/igdn_2/cond_1/cond/Equal:z:0Omodel_synthesis_layer_2_igdn_2_cond_1_cond_cond_model_synthesis_layer_2_biasadd2model_synthesis_layer_2_igdn_2_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *O
else_branch@R>
<model_synthesis_layer_2_igdn_2_cond_1_cond_cond_false_200387*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_2_igdn_2_cond_1_cond_cond_true_20038621
/model/synthesis/layer_2/igdn_2/cond_1/cond/cond?
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/IdentityIdentity8model/synthesis/layer_2/igdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8model/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity?
3model/synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentityAmodel/synthesis/layer_2/igdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_1/cond/Identity"s
3model_synthesis_layer_2_igdn_2_cond_1_cond_identity<model/synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
??
?
E__inference_synthesis_layer_call_and_return_conditional_losses_203663

inputs
layer_0_kernel_matmul_aA
-layer_0_kernel_matmul_readvariableop_resource:
??6
'layer_0_biasadd_readvariableop_resource:	?
layer_0_igdn_0_equal_xL
8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource:
??*
&layer_0_igdn_0_gamma_lower_bound_bound
layer_0_igdn_0_gamma_sub_yF
7layer_0_igdn_0_beta_lower_bound_readvariableop_resource:	?)
%layer_0_igdn_0_beta_lower_bound_bound
layer_0_igdn_0_beta_sub_y
layer_0_igdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??6
'layer_1_biasadd_readvariableop_resource:	?
layer_1_igdn_1_equal_xL
8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource:
??*
&layer_1_igdn_1_gamma_lower_bound_bound
layer_1_igdn_1_gamma_sub_yF
7layer_1_igdn_1_beta_lower_bound_readvariableop_resource:	?)
%layer_1_igdn_1_beta_lower_bound_bound
layer_1_igdn_1_beta_sub_y
layer_1_igdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??6
'layer_2_biasadd_readvariableop_resource:	?
layer_2_igdn_2_equal_xL
8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource:
??*
&layer_2_igdn_2_gamma_lower_bound_bound
layer_2_igdn_2_gamma_sub_yF
7layer_2_igdn_2_beta_lower_bound_readvariableop_resource:	?)
%layer_2_igdn_2_beta_lower_bound_bound
layer_2_igdn_2_beta_sub_y
layer_2_igdn_2_equal_1_x
layer_3_kernel_matmul_a@
-layer_3_kernel_matmul_readvariableop_resource:	?5
'layer_3_biasadd_readvariableop_resource:
identity??layer_0/BiasAdd/ReadVariableOp?.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?layer_1/BiasAdd/ReadVariableOp?.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/kernel/Reshape?
layer_0/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_0/transpose/perm?
layer_0/transpose	Transposelayer_0/kernel/Reshape:output:0layer_0/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_0/transposeT
layer_0/ShapeShapeinputs*
T0*
_output_shapes
:2
layer_0/Shape?
layer_0/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_0/strided_slice/stack?
layer_0/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice/stack_1?
layer_0/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice/stack_2?
layer_0/strided_sliceStridedSlicelayer_0/Shape:output:0$layer_0/strided_slice/stack:output:0&layer_0/strided_slice/stack_1:output:0&layer_0/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice?
layer_0/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice_1/stack?
layer_0/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_1/stack_1?
layer_0/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_1/stack_2?
layer_0/strided_slice_1StridedSlicelayer_0/Shape:output:0&layer_0/strided_slice_1/stack:output:0(layer_0/strided_slice_1/stack_1:output:0(layer_0/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice_1`
layer_0/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_0/mul/y|
layer_0/mulMul layer_0/strided_slice_1:output:0layer_0/mul/y:output:0*
T0*
_output_shapes
: 2
layer_0/mul`
layer_0/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_0/add/ym
layer_0/addAddV2layer_0/mul:z:0layer_0/add/y:output:0*
T0*
_output_shapes
: 2
layer_0/add?
layer_0/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_0/strided_slice_2/stack?
layer_0/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_2/stack_1?
layer_0/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_0/strided_slice_2/stack_2?
layer_0/strided_slice_2StridedSlicelayer_0/Shape:output:0&layer_0/strided_slice_2/stack:output:0(layer_0/strided_slice_2/stack_1:output:0(layer_0/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_0/strided_slice_2d
layer_0/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_0/mul_1/y?
layer_0/mul_1Mul layer_0/strided_slice_2:output:0layer_0/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_0/mul_1d
layer_0/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_0/add_1/yu
layer_0/add_1AddV2layer_0/mul_1:z:0layer_0/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_0/add_1?
&layer_0/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_0/conv2d_transpose/input_sizes/3?
$layer_0/conv2d_transpose/input_sizesPacklayer_0/strided_slice:output:0layer_0/add:z:0layer_0/add_1:z:0/layer_0/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_0/conv2d_transpose/input_sizes?
layer_0/conv2d_transposeConv2DBackpropInput-layer_0/conv2d_transpose/input_sizes:output:0layer_0/transpose:y:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_0/conv2d_transpose?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAdd!layer_0/conv2d_transpose:output:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/BiasAddi
layer_0/igdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/x?
layer_0/igdn_0/EqualEquallayer_0_igdn_0_equal_xlayer_0/igdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/Equal?
layer_0/igdn_0/condStatelessIflayer_0/igdn_0/Equal:z:0layer_0/igdn_0/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_0_igdn_0_cond_false_203181*
output_shapes
: *2
then_branch#R!
layer_0_igdn_0_cond_true_2031802
layer_0/igdn_0/cond?
layer_0/igdn_0/cond/IdentityIdentitylayer_0/igdn_0/cond:output:0*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity?
layer_0/igdn_0/cond_1StatelessIf%layer_0/igdn_0/cond/Identity:output:0layer_0/BiasAdd:output:0layer_0_igdn_0_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_0_igdn_0_cond_1_false_203192*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_1_true_2031912
layer_0/igdn_0/cond_1?
layer_0/igdn_0/cond_1/IdentityIdentitylayer_0/igdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity?
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp?
 layer_0/igdn_0/gamma/lower_boundMaximum7layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_0/igdn_0/gamma/lower_bound?
)layer_0/igdn_0/gamma/lower_bound/IdentityIdentity$layer_0/igdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_0/igdn_0/gamma/lower_bound/Identity?
*layer_0/igdn_0/gamma/lower_bound/IdentityN	IdentityN$layer_0/igdn_0/gamma/lower_bound:z:07layer_0/igdn_0/gamma/lower_bound/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203237*.
_output_shapes
:
??:
??: 2,
*layer_0/igdn_0/gamma/lower_bound/IdentityN?
layer_0/igdn_0/gamma/SquareSquare3layer_0/igdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square?
layer_0/igdn_0/gamma/subSublayer_0/igdn_0/gamma/Square:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub?
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_0_igdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp?
"layer_0/igdn_0/gamma/lower_bound_1Maximum9layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_0/igdn_0/gamma/lower_bound_1?
+layer_0/igdn_0/gamma/lower_bound_1/IdentityIdentity&layer_0/igdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_0/igdn_0/gamma/lower_bound_1/Identity?
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN	IdentityN&layer_0/igdn_0/gamma/lower_bound_1:z:09layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp:value:0&layer_0_igdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203247*.
_output_shapes
:
??:
??: 2.
,layer_0/igdn_0/gamma/lower_bound_1/IdentityN?
layer_0/igdn_0/gamma/Square_1Square5layer_0/igdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/Square_1?
layer_0/igdn_0/gamma/sub_1Sub!layer_0/igdn_0/gamma/Square_1:y:0layer_0_igdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/igdn_0/gamma/sub_1?
layer_0/igdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/igdn_0/Reshape/shape?
layer_0/igdn_0/ReshapeReshapelayer_0/igdn_0/gamma/sub_1:z:0%layer_0/igdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/igdn_0/Reshape?
layer_0/igdn_0/convolutionConv2D'layer_0/igdn_0/cond_1/Identity:output:0layer_0/igdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_0/igdn_0/convolution?
.layer_0/igdn_0/beta/lower_bound/ReadVariableOpReadVariableOp7layer_0_igdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp?
layer_0/igdn_0/beta/lower_boundMaximum6layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_0/igdn_0/beta/lower_bound?
(layer_0/igdn_0/beta/lower_bound/IdentityIdentity#layer_0/igdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_0/igdn_0/beta/lower_bound/Identity?
)layer_0/igdn_0/beta/lower_bound/IdentityN	IdentityN#layer_0/igdn_0/beta/lower_bound:z:06layer_0/igdn_0/beta/lower_bound/ReadVariableOp:value:0%layer_0_igdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203261*$
_output_shapes
:?:?: 2+
)layer_0/igdn_0/beta/lower_bound/IdentityN?
layer_0/igdn_0/beta/SquareSquare2layer_0/igdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/Square?
layer_0/igdn_0/beta/subSublayer_0/igdn_0/beta/Square:y:0layer_0_igdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/igdn_0/beta/sub?
layer_0/igdn_0/BiasAddBiasAdd#layer_0/igdn_0/convolution:output:0layer_0/igdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/igdn_0/BiasAddm
layer_0/igdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/igdn_0/x_1?
layer_0/igdn_0/Equal_1Equallayer_0_igdn_0_equal_1_xlayer_0/igdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/Equal_1?
layer_0/igdn_0/cond_2StatelessIflayer_0/igdn_0/Equal_1:z:0layer_0/igdn_0/BiasAdd:output:0layer_0_igdn_0_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_0_igdn_0_cond_2_false_203275*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_2_true_2032742
layer_0/igdn_0/cond_2?
layer_0/igdn_0/cond_2/IdentityIdentitylayer_0/igdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity?
layer_0/igdn_0/mulMullayer_0/BiasAdd:output:0'layer_0/igdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/igdn_0/mul?
$layer_1/kernel/MatMul/ReadVariableOpReadVariableOp-layer_1_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_1/kernel/MatMul/ReadVariableOp?
layer_1/kernel/MatMulMatMullayer_1_kernel_matmul_a,layer_1/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_1/kernel/MatMul?
layer_1/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/kernel/Reshape/shape?
layer_1/kernel/ReshapeReshapelayer_1/kernel/MatMul:product:0%layer_1/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/kernel/Reshape?
layer_1/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_1/transpose/perm?
layer_1/transpose	Transposelayer_1/kernel/Reshape:output:0layer_1/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_1/transposed
layer_1/ShapeShapelayer_0/igdn_0/mul:z:0*
T0*
_output_shapes
:2
layer_1/Shape?
layer_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_1/strided_slice/stack?
layer_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice/stack_1?
layer_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice/stack_2?
layer_1/strided_sliceStridedSlicelayer_1/Shape:output:0$layer_1/strided_slice/stack:output:0&layer_1/strided_slice/stack_1:output:0&layer_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice?
layer_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice_1/stack?
layer_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_1/stack_1?
layer_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_1/stack_2?
layer_1/strided_slice_1StridedSlicelayer_1/Shape:output:0&layer_1/strided_slice_1/stack:output:0(layer_1/strided_slice_1/stack_1:output:0(layer_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice_1`
layer_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_1/mul/y|
layer_1/mulMul layer_1/strided_slice_1:output:0layer_1/mul/y:output:0*
T0*
_output_shapes
: 2
layer_1/mul`
layer_1/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_1/add/ym
layer_1/addAddV2layer_1/mul:z:0layer_1/add/y:output:0*
T0*
_output_shapes
: 2
layer_1/add?
layer_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_1/strided_slice_2/stack?
layer_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_2/stack_1?
layer_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_1/strided_slice_2/stack_2?
layer_1/strided_slice_2StridedSlicelayer_1/Shape:output:0&layer_1/strided_slice_2/stack:output:0(layer_1/strided_slice_2/stack_1:output:0(layer_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_1/strided_slice_2d
layer_1/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_1/mul_1/y?
layer_1/mul_1Mul layer_1/strided_slice_2:output:0layer_1/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_1/mul_1d
layer_1/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_1/add_1/yu
layer_1/add_1AddV2layer_1/mul_1:z:0layer_1/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_1/add_1?
&layer_1/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_1/conv2d_transpose/input_sizes/3?
$layer_1/conv2d_transpose/input_sizesPacklayer_1/strided_slice:output:0layer_1/add:z:0layer_1/add_1:z:0/layer_1/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_1/conv2d_transpose/input_sizes?
layer_1/conv2d_transposeConv2DBackpropInput-layer_1/conv2d_transpose/input_sizes:output:0layer_1/transpose:y:0layer_0/igdn_0/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_1/conv2d_transpose?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAdd!layer_1/conv2d_transpose:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/BiasAddi
layer_1/igdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/x?
layer_1/igdn_1/EqualEquallayer_1_igdn_1_equal_xlayer_1/igdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/Equal?
layer_1/igdn_1/condStatelessIflayer_1/igdn_1/Equal:z:0layer_1/igdn_1/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_1_igdn_1_cond_false_203342*
output_shapes
: *2
then_branch#R!
layer_1_igdn_1_cond_true_2033412
layer_1/igdn_1/cond?
layer_1/igdn_1/cond/IdentityIdentitylayer_1/igdn_1/cond:output:0*
T0
*
_output_shapes
: 2
layer_1/igdn_1/cond/Identity?
layer_1/igdn_1/cond_1StatelessIf%layer_1/igdn_1/cond/Identity:output:0layer_1/BiasAdd:output:0layer_1_igdn_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_1_igdn_1_cond_1_false_203353*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_1_true_2033522
layer_1/igdn_1/cond_1?
layer_1/igdn_1/cond_1/IdentityIdentitylayer_1/igdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_1/Identity?
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp?
 layer_1/igdn_1/gamma/lower_boundMaximum7layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_1/igdn_1/gamma/lower_bound?
)layer_1/igdn_1/gamma/lower_bound/IdentityIdentity$layer_1/igdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_1/igdn_1/gamma/lower_bound/Identity?
*layer_1/igdn_1/gamma/lower_bound/IdentityN	IdentityN$layer_1/igdn_1/gamma/lower_bound:z:07layer_1/igdn_1/gamma/lower_bound/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203398*.
_output_shapes
:
??:
??: 2,
*layer_1/igdn_1/gamma/lower_bound/IdentityN?
layer_1/igdn_1/gamma/SquareSquare3layer_1/igdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square?
layer_1/igdn_1/gamma/subSublayer_1/igdn_1/gamma/Square:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub?
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_1_igdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp?
"layer_1/igdn_1/gamma/lower_bound_1Maximum9layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_1/igdn_1/gamma/lower_bound_1?
+layer_1/igdn_1/gamma/lower_bound_1/IdentityIdentity&layer_1/igdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_1/igdn_1/gamma/lower_bound_1/Identity?
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN	IdentityN&layer_1/igdn_1/gamma/lower_bound_1:z:09layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp:value:0&layer_1_igdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203408*.
_output_shapes
:
??:
??: 2.
,layer_1/igdn_1/gamma/lower_bound_1/IdentityN?
layer_1/igdn_1/gamma/Square_1Square5layer_1/igdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/Square_1?
layer_1/igdn_1/gamma/sub_1Sub!layer_1/igdn_1/gamma/Square_1:y:0layer_1_igdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/igdn_1/gamma/sub_1?
layer_1/igdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/igdn_1/Reshape/shape?
layer_1/igdn_1/ReshapeReshapelayer_1/igdn_1/gamma/sub_1:z:0%layer_1/igdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/igdn_1/Reshape?
layer_1/igdn_1/convolutionConv2D'layer_1/igdn_1/cond_1/Identity:output:0layer_1/igdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_1/igdn_1/convolution?
.layer_1/igdn_1/beta/lower_bound/ReadVariableOpReadVariableOp7layer_1_igdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp?
layer_1/igdn_1/beta/lower_boundMaximum6layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_1/igdn_1/beta/lower_bound?
(layer_1/igdn_1/beta/lower_bound/IdentityIdentity#layer_1/igdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_1/igdn_1/beta/lower_bound/Identity?
)layer_1/igdn_1/beta/lower_bound/IdentityN	IdentityN#layer_1/igdn_1/beta/lower_bound:z:06layer_1/igdn_1/beta/lower_bound/ReadVariableOp:value:0%layer_1_igdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203422*$
_output_shapes
:?:?: 2+
)layer_1/igdn_1/beta/lower_bound/IdentityN?
layer_1/igdn_1/beta/SquareSquare2layer_1/igdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/Square?
layer_1/igdn_1/beta/subSublayer_1/igdn_1/beta/Square:y:0layer_1_igdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/igdn_1/beta/sub?
layer_1/igdn_1/BiasAddBiasAdd#layer_1/igdn_1/convolution:output:0layer_1/igdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/igdn_1/BiasAddm
layer_1/igdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/igdn_1/x_1?
layer_1/igdn_1/Equal_1Equallayer_1_igdn_1_equal_1_xlayer_1/igdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/igdn_1/Equal_1?
layer_1/igdn_1/cond_2StatelessIflayer_1/igdn_1/Equal_1:z:0layer_1/igdn_1/BiasAdd:output:0layer_1_igdn_1_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_1_igdn_1_cond_2_false_203436*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_2_true_2034352
layer_1/igdn_1/cond_2?
layer_1/igdn_1/cond_2/IdentityIdentitylayer_1/igdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity?
layer_1/igdn_1/mulMullayer_1/BiasAdd:output:0'layer_1/igdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/igdn_1/mul?
$layer_2/kernel/MatMul/ReadVariableOpReadVariableOp-layer_2_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_2/kernel/MatMul/ReadVariableOp?
layer_2/kernel/MatMulMatMullayer_2_kernel_matmul_a,layer_2/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_2/kernel/MatMul?
layer_2/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/kernel/Reshape/shape?
layer_2/kernel/ReshapeReshapelayer_2/kernel/MatMul:product:0%layer_2/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/kernel/Reshape?
layer_2/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_2/transpose/perm?
layer_2/transpose	Transposelayer_2/kernel/Reshape:output:0layer_2/transpose/perm:output:0*
T0*(
_output_shapes
:??2
layer_2/transposed
layer_2/ShapeShapelayer_1/igdn_1/mul:z:0*
T0*
_output_shapes
:2
layer_2/Shape?
layer_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_2/strided_slice/stack?
layer_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice/stack_1?
layer_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice/stack_2?
layer_2/strided_sliceStridedSlicelayer_2/Shape:output:0$layer_2/strided_slice/stack:output:0&layer_2/strided_slice/stack_1:output:0&layer_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice?
layer_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice_1/stack?
layer_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_1/stack_1?
layer_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_1/stack_2?
layer_2/strided_slice_1StridedSlicelayer_2/Shape:output:0&layer_2/strided_slice_1/stack:output:0(layer_2/strided_slice_1/stack_1:output:0(layer_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice_1`
layer_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_2/mul/y|
layer_2/mulMul layer_2/strided_slice_1:output:0layer_2/mul/y:output:0*
T0*
_output_shapes
: 2
layer_2/mul`
layer_2/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_2/add/ym
layer_2/addAddV2layer_2/mul:z:0layer_2/add/y:output:0*
T0*
_output_shapes
: 2
layer_2/add?
layer_2/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_2/strided_slice_2/stack?
layer_2/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_2/stack_1?
layer_2/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_2/strided_slice_2/stack_2?
layer_2/strided_slice_2StridedSlicelayer_2/Shape:output:0&layer_2/strided_slice_2/stack:output:0(layer_2/strided_slice_2/stack_1:output:0(layer_2/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_2/strided_slice_2d
layer_2/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_2/mul_1/y?
layer_2/mul_1Mul layer_2/strided_slice_2:output:0layer_2/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_2/mul_1d
layer_2/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_2/add_1/yu
layer_2/add_1AddV2layer_2/mul_1:z:0layer_2/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_2/add_1?
&layer_2/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value
B :?2(
&layer_2/conv2d_transpose/input_sizes/3?
$layer_2/conv2d_transpose/input_sizesPacklayer_2/strided_slice:output:0layer_2/add:z:0layer_2/add_1:z:0/layer_2/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_2/conv2d_transpose/input_sizes?
layer_2/conv2d_transposeConv2DBackpropInput-layer_2/conv2d_transpose/input_sizes:output:0layer_2/transpose:y:0layer_1/igdn_1/mul:z:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_2/conv2d_transpose?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAdd!layer_2/conv2d_transpose:output:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/BiasAddi
layer_2/igdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/x?
layer_2/igdn_2/EqualEquallayer_2_igdn_2_equal_xlayer_2/igdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/Equal?
layer_2/igdn_2/condStatelessIflayer_2/igdn_2/Equal:z:0layer_2/igdn_2/Equal:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *3
else_branch$R"
 layer_2_igdn_2_cond_false_203503*
output_shapes
: *2
then_branch#R!
layer_2_igdn_2_cond_true_2035022
layer_2/igdn_2/cond?
layer_2/igdn_2/cond/IdentityIdentitylayer_2/igdn_2/cond:output:0*
T0
*
_output_shapes
: 2
layer_2/igdn_2/cond/Identity?
layer_2/igdn_2/cond_1StatelessIf%layer_2/igdn_2/cond/Identity:output:0layer_2/BiasAdd:output:0layer_2_igdn_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_2_igdn_2_cond_1_false_203514*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_1_true_2035132
layer_2/igdn_2/cond_1?
layer_2/igdn_2/cond_1/IdentityIdentitylayer_2/igdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/Identity?
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype021
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp?
 layer_2/igdn_2/gamma/lower_boundMaximum7layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2"
 layer_2/igdn_2/gamma/lower_bound?
)layer_2/igdn_2/gamma/lower_bound/IdentityIdentity$layer_2/igdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2+
)layer_2/igdn_2/gamma/lower_bound/Identity?
*layer_2/igdn_2/gamma/lower_bound/IdentityN	IdentityN$layer_2/igdn_2/gamma/lower_bound:z:07layer_2/igdn_2/gamma/lower_bound/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203559*.
_output_shapes
:
??:
??: 2,
*layer_2/igdn_2/gamma/lower_bound/IdentityN?
layer_2/igdn_2/gamma/SquareSquare3layer_2/igdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square?
layer_2/igdn_2/gamma/subSublayer_2/igdn_2/gamma/Square:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub?
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp8layer_2_igdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype023
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp?
"layer_2/igdn_2/gamma/lower_bound_1Maximum9layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2$
"layer_2/igdn_2/gamma/lower_bound_1?
+layer_2/igdn_2/gamma/lower_bound_1/IdentityIdentity&layer_2/igdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2-
+layer_2/igdn_2/gamma/lower_bound_1/Identity?
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN	IdentityN&layer_2/igdn_2/gamma/lower_bound_1:z:09layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp:value:0&layer_2_igdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203569*.
_output_shapes
:
??:
??: 2.
,layer_2/igdn_2/gamma/lower_bound_1/IdentityN?
layer_2/igdn_2/gamma/Square_1Square5layer_2/igdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/Square_1?
layer_2/igdn_2/gamma/sub_1Sub!layer_2/igdn_2/gamma/Square_1:y:0layer_2_igdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/igdn_2/gamma/sub_1?
layer_2/igdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/igdn_2/Reshape/shape?
layer_2/igdn_2/ReshapeReshapelayer_2/igdn_2/gamma/sub_1:z:0%layer_2/igdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/igdn_2/Reshape?
layer_2/igdn_2/convolutionConv2D'layer_2/igdn_2/cond_1/Identity:output:0layer_2/igdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_2/igdn_2/convolution?
.layer_2/igdn_2/beta/lower_bound/ReadVariableOpReadVariableOp7layer_2_igdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype020
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp?
layer_2/igdn_2/beta/lower_boundMaximum6layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2!
layer_2/igdn_2/beta/lower_bound?
(layer_2/igdn_2/beta/lower_bound/IdentityIdentity#layer_2/igdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2*
(layer_2/igdn_2/beta/lower_bound/Identity?
)layer_2/igdn_2/beta/lower_bound/IdentityN	IdentityN#layer_2/igdn_2/beta/lower_bound:z:06layer_2/igdn_2/beta/lower_bound/ReadVariableOp:value:0%layer_2_igdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-203583*$
_output_shapes
:?:?: 2+
)layer_2/igdn_2/beta/lower_bound/IdentityN?
layer_2/igdn_2/beta/SquareSquare2layer_2/igdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/Square?
layer_2/igdn_2/beta/subSublayer_2/igdn_2/beta/Square:y:0layer_2_igdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/igdn_2/beta/sub?
layer_2/igdn_2/BiasAddBiasAdd#layer_2/igdn_2/convolution:output:0layer_2/igdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/igdn_2/BiasAddm
layer_2/igdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/igdn_2/x_1?
layer_2/igdn_2/Equal_1Equallayer_2_igdn_2_equal_1_xlayer_2/igdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/igdn_2/Equal_1?
layer_2/igdn_2/cond_2StatelessIflayer_2/igdn_2/Equal_1:z:0layer_2/igdn_2/BiasAdd:output:0layer_2_igdn_2_equal_1_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *5
else_branch&R$
"layer_2_igdn_2_cond_2_false_203597*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_2_true_2035962
layer_2/igdn_2/cond_2?
layer_2/igdn_2/cond_2/IdentityIdentitylayer_2/igdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_2/Identity?
layer_2/igdn_2/mulMullayer_2/BiasAdd:output:0'layer_2/igdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/igdn_2/mul?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?      2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_3/kernel/Reshape?
layer_3/transpose/permConst*
_output_shapes
:*
dtype0*%
valueB"             2
layer_3/transpose/perm?
layer_3/transpose	Transposelayer_3/kernel/Reshape:output:0layer_3/transpose/perm:output:0*
T0*'
_output_shapes
:?2
layer_3/transposed
layer_3/ShapeShapelayer_2/igdn_2/mul:z:0*
T0*
_output_shapes
:2
layer_3/Shape?
layer_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
layer_3/strided_slice/stack?
layer_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice/stack_1?
layer_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice/stack_2?
layer_3/strided_sliceStridedSlicelayer_3/Shape:output:0$layer_3/strided_slice/stack:output:0&layer_3/strided_slice/stack_1:output:0&layer_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice?
layer_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice_1/stack?
layer_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_1/stack_1?
layer_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_1/stack_2?
layer_3/strided_slice_1StridedSlicelayer_3/Shape:output:0&layer_3/strided_slice_1/stack:output:0(layer_3/strided_slice_1/stack_1:output:0(layer_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice_1`
layer_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_3/mul/y|
layer_3/mulMul layer_3/strided_slice_1:output:0layer_3/mul/y:output:0*
T0*
_output_shapes
: 2
layer_3/mul`
layer_3/add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_3/add/ym
layer_3/addAddV2layer_3/mul:z:0layer_3/add/y:output:0*
T0*
_output_shapes
: 2
layer_3/add?
layer_3/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
layer_3/strided_slice_2/stack?
layer_3/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_2/stack_1?
layer_3/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
layer_3/strided_slice_2/stack_2?
layer_3/strided_slice_2StridedSlicelayer_3/Shape:output:0&layer_3/strided_slice_2/stack:output:0(layer_3/strided_slice_2/stack_1:output:0(layer_3/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
layer_3/strided_slice_2d
layer_3/mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
layer_3/mul_1/y?
layer_3/mul_1Mul layer_3/strided_slice_2:output:0layer_3/mul_1/y:output:0*
T0*
_output_shapes
: 2
layer_3/mul_1d
layer_3/add_1/yConst*
_output_shapes
: *
dtype0*
value	B : 2
layer_3/add_1/yu
layer_3/add_1AddV2layer_3/mul_1:z:0layer_3/add_1/y:output:0*
T0*
_output_shapes
: 2
layer_3/add_1?
&layer_3/conv2d_transpose/input_sizes/3Const*
_output_shapes
: *
dtype0*
value	B :2(
&layer_3/conv2d_transpose/input_sizes/3?
$layer_3/conv2d_transpose/input_sizesPacklayer_3/strided_slice:output:0layer_3/add:z:0layer_3/add_1:z:0/layer_3/conv2d_transpose/input_sizes/3:output:0*
N*
T0*
_output_shapes
:2&
$layer_3/conv2d_transpose/input_sizes?
layer_3/conv2d_transposeConv2DBackpropInput-layer_3/conv2d_transpose/input_sizes:output:0layer_3/transpose:y:0layer_2/igdn_2/mul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_3/conv2d_transpose?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAdd!layer_3/conv2d_transpose:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2
layer_3/BiasAdde
lambda_1/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
lambda_1/mul/y?
lambda_1/mulMullayer_3/BiasAdd:output:0lambda_1/mul/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda_1/mul?
IdentityIdentitylambda_1/mul:z:0^layer_0/BiasAdd/ReadVariableOp/^layer_0/igdn_0/beta/lower_bound/ReadVariableOp0^layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2^layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp/^layer_1/igdn_1/beta/lower_bound/ReadVariableOp0^layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2^layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp/^layer_2/igdn_2/beta/lower_bound/ReadVariableOp0^layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2^layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2@
layer_0/BiasAdd/ReadVariableOplayer_0/BiasAdd/ReadVariableOp2`
.layer_0/igdn_0/beta/lower_bound/ReadVariableOp.layer_0/igdn_0/beta/lower_bound/ReadVariableOp2b
/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp/layer_0/igdn_0/gamma/lower_bound/ReadVariableOp2f
1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp1layer_0/igdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2`
.layer_1/igdn_1/beta/lower_bound/ReadVariableOp.layer_1/igdn_1/beta/lower_bound/ReadVariableOp2b
/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp/layer_1/igdn_1/gamma/lower_bound/ReadVariableOp2f
1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp1layer_1/igdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2`
.layer_2/igdn_2/beta/lower_bound/ReadVariableOp.layer_2/igdn_2/beta/lower_bound/ReadVariableOp2b
/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp/layer_2/igdn_2/gamma/lower_bound/ReadVariableOp2f
1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp1layer_2/igdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
?
A__inference_model_layer_call_and_return_conditional_losses_201626
input_2
synthesis_201552$
synthesis_201554:
??
synthesis_201556:	?
synthesis_201558$
synthesis_201560:
??
synthesis_201562
synthesis_201564
synthesis_201566:	?
synthesis_201568
synthesis_201570
synthesis_201572
synthesis_201574$
synthesis_201576:
??
synthesis_201578:	?
synthesis_201580$
synthesis_201582:
??
synthesis_201584
synthesis_201586
synthesis_201588:	?
synthesis_201590
synthesis_201592
synthesis_201594
synthesis_201596$
synthesis_201598:
??
synthesis_201600:	?
synthesis_201602$
synthesis_201604:
??
synthesis_201606
synthesis_201608
synthesis_201610:	?
synthesis_201612
synthesis_201614
synthesis_201616
synthesis_201618#
synthesis_201620:	?
synthesis_201622:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinput_2synthesis_201552synthesis_201554synthesis_201556synthesis_201558synthesis_201560synthesis_201562synthesis_201564synthesis_201566synthesis_201568synthesis_201570synthesis_201572synthesis_201574synthesis_201576synthesis_201578synthesis_201580synthesis_201582synthesis_201584synthesis_201586synthesis_201588synthesis_201590synthesis_201592synthesis_201594synthesis_201596synthesis_201598synthesis_201600synthesis_201602synthesis_201604synthesis_201606synthesis_201608synthesis_201610synthesis_201612synthesis_201614synthesis_201616synthesis_201618synthesis_201620synthesis_201622*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_synthesis_layer_call_and_return_conditional_losses_2013152#
!synthesis/StatefulPartitionedCall?
IdentityIdentity*synthesis/StatefulPartitionedCall:output:0"^synthesis/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:,????????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2F
!synthesis/StatefulPartitionedCall!synthesis/StatefulPartitionedCall:k g
B
_output_shapes0
.:,????????????????????????????
!
_user_specified_name	input_2:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :$" 

_output_shapes

:
?
z
igdn_0_cond_1_true_204243"
igdn_0_cond_1_identity_biasadd
igdn_0_cond_1_placeholder
igdn_0_cond_1_identity?
igdn_0/cond_1/IdentityIdentityigdn_0_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/Identity"9
igdn_0_cond_1_identityigdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
P
igdn_1_cond_true_200751
igdn_1_cond_placeholder

igdn_1_cond_identity
h
igdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_1/cond/Constu
igdn_1/cond/IdentityIdentityigdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_1/cond/Identity"5
igdn_1_cond_identityigdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
h
layer_0_igdn_0_cond_true_203180#
layer_0_igdn_0_cond_placeholder
 
layer_0_igdn_0_cond_identity
x
layer_0/igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_0/igdn_0/cond/Const?
layer_0/igdn_0/cond/IdentityIdentity"layer_0/igdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_0/igdn_0/cond/Identity"E
layer_0_igdn_0_cond_identity%layer_0/igdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
,layer_0_igdn_0_cond_1_cond_cond_false_2032117
3layer_0_igdn_0_cond_1_cond_cond_pow_layer_0_biasadd)
%layer_0_igdn_0_cond_1_cond_cond_pow_y,
(layer_0_igdn_0_cond_1_cond_cond_identity?
#layer_0/igdn_0/cond_1/cond/cond/powPow3layer_0_igdn_0_cond_1_cond_cond_pow_layer_0_biasadd%layer_0_igdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_1/cond/cond/pow?
(layer_0/igdn_0/cond_1/cond/cond/IdentityIdentity'layer_0/igdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_0/igdn_0/cond_1/cond/cond/Identity"]
(layer_0_igdn_0_cond_1_cond_cond_identity1layer_0/igdn_0/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_1_igdn_1_cond_2_cond_true_203968:
6layer_1_igdn_1_cond_2_cond_sqrt_layer_1_igdn_1_biasadd*
&layer_1_igdn_1_cond_2_cond_placeholder'
#layer_1_igdn_1_cond_2_cond_identity?
layer_1/igdn_1/cond_2/cond/SqrtSqrt6layer_1_igdn_1_cond_2_cond_sqrt_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2!
layer_1/igdn_1/cond_2/cond/Sqrt?
#layer_1/igdn_1/cond_2/cond/IdentityIdentity#layer_1/igdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_2/cond/Identity"S
#layer_1_igdn_1_cond_2_cond_identity,layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_2_cond_true_200854*
&igdn_1_cond_2_cond_sqrt_igdn_1_biasadd"
igdn_1_cond_2_cond_placeholder
igdn_1_cond_2_cond_identity?
igdn_1/cond_2/cond/SqrtSqrt&igdn_1_cond_2_cond_sqrt_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Sqrt?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity"C
igdn_1_cond_2_cond_identity$igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
&layer_2_igdn_2_cond_1_cond_true_2035222
.layer_2_igdn_2_cond_1_cond_abs_layer_2_biasadd*
&layer_2_igdn_2_cond_1_cond_placeholder'
#layer_2_igdn_2_cond_1_cond_identity?
layer_2/igdn_2/cond_1/cond/AbsAbs.layer_2_igdn_2_cond_1_cond_abs_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/igdn_2/cond_1/cond/Abs?
#layer_2/igdn_2/cond_1/cond/IdentityIdentity"layer_2/igdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_2/igdn_2/cond_1/cond/Identity"S
#layer_2_igdn_2_cond_1_cond_identity,layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
s
igdn_2_cond_1_false_200952
igdn_2_cond_1_cond_biasadd
igdn_2_cond_1_equal_x
igdn_2_cond_1_identityg
igdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
igdn_2/cond_1/x?
igdn_2/cond_1/EqualEqualigdn_2_cond_1_equal_xigdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_2/cond_1/Equal?
igdn_2/cond_1/condStatelessIfigdn_2/cond_1/Equal:z:0igdn_2_cond_1_cond_biasaddigdn_2_cond_1_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
else_branch#R!
igdn_2_cond_1_cond_false_200961*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_1_cond_true_2009602
igdn_2/cond_1/cond?
igdn_2/cond_1/cond/IdentityIdentityigdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/cond/Identity?
igdn_2/cond_1/IdentityIdentity$igdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_2/cond_1/Identity"9
igdn_2_cond_1_identityigdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
"layer_0_igdn_0_cond_2_false_2037995
1layer_0_igdn_0_cond_2_cond_layer_0_igdn_0_biasadd!
layer_0_igdn_0_cond_2_equal_x"
layer_0_igdn_0_cond_2_identityw
layer_0/igdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_0/igdn_0/cond_2/x?
layer_0/igdn_0/cond_2/EqualEquallayer_0_igdn_0_cond_2_equal_x layer_0/igdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/igdn_0/cond_2/Equal?
layer_0/igdn_0/cond_2/condStatelessIflayer_0/igdn_0/cond_2/Equal:z:01layer_0_igdn_0_cond_2_cond_layer_0_igdn_0_biasaddlayer_0_igdn_0_cond_2_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *:
else_branch+R)
'layer_0_igdn_0_cond_2_cond_false_203808*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_2_cond_true_2038072
layer_0/igdn_0/cond_2/cond?
#layer_0/igdn_0/cond_2/cond/IdentityIdentity#layer_0/igdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_0/igdn_0/cond_2/cond/Identity?
layer_0/igdn_0/cond_2/IdentityIdentity,layer_0/igdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_2/Identity"I
layer_0_igdn_0_cond_2_identity'layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
,layer_1_igdn_1_cond_1_cond_cond_false_2038967
3layer_1_igdn_1_cond_1_cond_cond_pow_layer_1_biasadd)
%layer_1_igdn_1_cond_1_cond_cond_pow_y,
(layer_1_igdn_1_cond_1_cond_cond_identity?
#layer_1/igdn_1/cond_1/cond/cond/powPow3layer_1_igdn_1_cond_1_cond_cond_pow_layer_1_biasadd%layer_1_igdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2%
#layer_1/igdn_1/cond_1/cond/cond/pow?
(layer_1/igdn_1/cond_1/cond/cond/IdentityIdentity'layer_1/igdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2*
(layer_1/igdn_1/cond_1/cond/cond/Identity"]
(layer_1_igdn_1_cond_1_cond_cond_identity1layer_1/igdn_1/cond_1/cond/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
??
?	
"__inference__traced_restore_204918
file_prefix,
assignvariableop_layer_0_bias:	?=
.assignvariableop_1_layer_0_igdn_0_reparam_beta:	?C
/assignvariableop_2_layer_0_igdn_0_reparam_gamma:
??:
&assignvariableop_3_layer_0_kernel_rdft:
??.
assignvariableop_4_layer_1_bias:	?=
.assignvariableop_5_layer_1_igdn_1_reparam_beta:	?C
/assignvariableop_6_layer_1_igdn_1_reparam_gamma:
??:
&assignvariableop_7_layer_1_kernel_rdft:
??.
assignvariableop_8_layer_2_bias:	?=
.assignvariableop_9_layer_2_igdn_2_reparam_beta:	?D
0assignvariableop_10_layer_2_igdn_2_reparam_gamma:
??;
'assignvariableop_11_layer_2_kernel_rdft:
??.
 assignvariableop_12_layer_3_bias::
'assignvariableop_13_layer_3_kernel_rdft:	?
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_layer_0_biasIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp.assignvariableop_1_layer_0_igdn_0_reparam_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp/assignvariableop_2_layer_0_igdn_0_reparam_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp&assignvariableop_3_layer_0_kernel_rdftIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_layer_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_layer_1_igdn_1_reparam_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp/assignvariableop_6_layer_1_igdn_1_reparam_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_layer_1_kernel_rdftIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_layer_2_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_layer_2_igdn_2_reparam_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_layer_2_igdn_2_reparam_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp'assignvariableop_11_layer_2_kernel_rdftIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp assignvariableop_12_layer_3_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp'assignvariableop_13_layer_3_kernel_rdftIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
?
?
*synthesis_layer_1_igdn_1_cond_false_202294I
Esynthesis_layer_1_igdn_1_cond_identity_synthesis_layer_1_igdn_1_equal
*
&synthesis_layer_1_igdn_1_cond_identity
?
&synthesis/layer_1/igdn_1/cond/IdentityIdentityEsynthesis_layer_1_igdn_1_cond_identity_synthesis_layer_1_igdn_1_equal*
T0
*
_output_shapes
: 2(
&synthesis/layer_1/igdn_1/cond/Identity"Y
&synthesis_layer_1_igdn_1_cond_identity/synthesis/layer_1/igdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+synthesis_layer_2_igdn_2_cond_2_true_203072M
Isynthesis_layer_2_igdn_2_cond_2_identity_synthesis_layer_2_igdn_2_biasadd/
+synthesis_layer_2_igdn_2_cond_2_placeholder,
(synthesis_layer_2_igdn_2_cond_2_identity?
(synthesis/layer_2/igdn_2/cond_2/IdentityIdentityIsynthesis_layer_2_igdn_2_cond_2_identity_synthesis_layer_2_igdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_2/igdn_2/cond_2/Identity"]
(synthesis_layer_2_igdn_2_cond_2_identity1synthesis/layer_2/igdn_2/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
6model_synthesis_layer_2_igdn_2_cond_1_cond_true_200376R
Nmodel_synthesis_layer_2_igdn_2_cond_1_cond_abs_model_synthesis_layer_2_biasadd:
6model_synthesis_layer_2_igdn_2_cond_1_cond_placeholder7
3model_synthesis_layer_2_igdn_2_cond_1_cond_identity?
.model/synthesis/layer_2/igdn_2/cond_1/cond/AbsAbsNmodel_synthesis_layer_2_igdn_2_cond_1_cond_abs_model_synthesis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.model/synthesis/layer_2/igdn_2/cond_1/cond/Abs?
3model/synthesis/layer_2/igdn_2/cond_1/cond/IdentityIdentity2model/synthesis/layer_2/igdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3model/synthesis/layer_2/igdn_2/cond_1/cond/Identity"s
3model_synthesis_layer_2_igdn_2_cond_1_cond_identity<model/synthesis/layer_2/igdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_0_igdn_0_cond_1_true_2031912
.layer_0_igdn_0_cond_1_identity_layer_0_biasadd%
!layer_0_igdn_0_cond_1_placeholder"
layer_0_igdn_0_cond_1_identity?
layer_0/igdn_0/cond_1/IdentityIdentity.layer_0_igdn_0_cond_1_identity_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/igdn_0/cond_1/Identity"I
layer_0_igdn_0_cond_1_identity'layer_0/igdn_0/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
!layer_1_igdn_1_cond_2_true_2034359
5layer_1_igdn_1_cond_2_identity_layer_1_igdn_1_biasadd%
!layer_1_igdn_1_cond_2_placeholder"
layer_1_igdn_1_cond_2_identity?
layer_1/igdn_1/cond_2/IdentityIdentity5layer_1_igdn_1_cond_2_identity_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/igdn_1/cond_2/Identity"I
layer_1_igdn_1_cond_2_identity'layer_1/igdn_1/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
igdn_1_cond_2_cond_false_204505)
%igdn_1_cond_2_cond_pow_igdn_1_biasadd
igdn_1_cond_2_cond_pow_y
igdn_1_cond_2_cond_identity?
igdn_1/cond_2/cond/powPow%igdn_1_cond_2_cond_pow_igdn_1_biasaddigdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/pow?
igdn_1/cond_2/cond/IdentityIdentityigdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_1/cond_2/cond/Identity"C
igdn_1_cond_2_cond_identity$igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
+synthesis_layer_0_igdn_0_cond_2_true_202750M
Isynthesis_layer_0_igdn_0_cond_2_identity_synthesis_layer_0_igdn_0_biasadd/
+synthesis_layer_0_igdn_0_cond_2_placeholder,
(synthesis_layer_0_igdn_0_cond_2_identity?
(synthesis/layer_0/igdn_0/cond_2/IdentityIdentityIsynthesis_layer_0_igdn_0_cond_2_identity_synthesis_layer_0_igdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2*
(synthesis/layer_0/igdn_0/cond_2/Identity"]
(synthesis_layer_0_igdn_0_cond_2_identity1synthesis/layer_0/igdn_0/cond_2/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
0synthesis_layer_1_igdn_1_cond_2_cond_true_202920N
Jsynthesis_layer_1_igdn_1_cond_2_cond_sqrt_synthesis_layer_1_igdn_1_biasadd4
0synthesis_layer_1_igdn_1_cond_2_cond_placeholder1
-synthesis_layer_1_igdn_1_cond_2_cond_identity?
)synthesis/layer_1/igdn_1/cond_2/cond/SqrtSqrtJsynthesis_layer_1_igdn_1_cond_2_cond_sqrt_synthesis_layer_1_igdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2+
)synthesis/layer_1/igdn_1/cond_2/cond/Sqrt?
-synthesis/layer_1/igdn_1/cond_2/cond/IdentityIdentity-synthesis/layer_1/igdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2/
-synthesis/layer_1/igdn_1/cond_2/cond/Identity"g
-synthesis_layer_1_igdn_1_cond_2_cond_identity6synthesis/layer_1/igdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
P
igdn_0_cond_true_204232
igdn_0_cond_placeholder

igdn_0_cond_identity
h
igdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
igdn_0/cond/Constu
igdn_0/cond/IdentityIdentityigdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
igdn_0/cond/Identity"5
igdn_0_cond_identityigdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
igdn_0_cond_1_cond_false_204253#
igdn_0_cond_1_cond_cond_biasadd
igdn_0_cond_1_cond_equal_x
igdn_0_cond_1_cond_identityq
igdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
igdn_0/cond_1/cond/x?
igdn_0/cond_1/cond/EqualEqualigdn_0_cond_1_cond_equal_xigdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
igdn_0/cond_1/cond/Equal?
igdn_0/cond_1/cond/condStatelessIfigdn_0/cond_1/cond/Equal:z:0igdn_0_cond_1_cond_cond_biasaddigdn_0_cond_1_cond_equal_x*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *7
else_branch(R&
$igdn_0_cond_1_cond_cond_false_204263*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_0_cond_1_cond_cond_true_2042622
igdn_0/cond_1/cond/cond?
 igdn_0/cond_1/cond/cond/IdentityIdentity igdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 igdn_0/cond_1/cond/cond/Identity?
igdn_0/cond_1/cond/IdentityIdentity)igdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
igdn_0/cond_1/cond/Identity"C
igdn_0_cond_1_cond_identity$igdn_0/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
V
input_2K
serving_default_input_2:0,????????????????????????????W
	synthesisJ
StatefulPartitionedCall:0+???????????????????????????tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*~&call_and_return_all_conditional_losses
_default_save_signature
?__call__"??
_tf_keras_network??{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "name": "synthesis", "inbound_nodes": [[["input_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["synthesis", 1, 0]]}, "shared_object_id": 39, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 192]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "name": "synthesis", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 38}], "input_layers": [["input_2", 0, 0]], "output_layers": [["synthesis", 1, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
??
layer_with_weights-0
layer-0
	layer_with_weights-1
	layer-1

layer_with_weights-2

layer-2
layer_with_weights-3
layer-3
layer-4
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_sequential??{"name": "synthesis", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 192]}, "float32", "layer_0_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}, "shared_object_id": 1}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 2}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 12}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 13}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 22}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 23}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 32}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}, "shared_object_id": 33}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 36}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 37}]}}}
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
?

layers
 layer_regularization_losses
!non_trainable_variables
trainable_variables
"metrics
regularization_losses
	variables
#layer_metrics
?__call__
_default_save_signature
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
$_activation
%_kernel_parameter
_bias_parameter
&trainable_variables
'regularization_losses
(	variables
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 2}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
*_activation
+_kernel_parameter
_bias_parameter
,trainable_variables
-regularization_losses
.	variables
/	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 13}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 22, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
0_activation
1_kernel_parameter
_bias_parameter
2trainable_variables
3regularization_losses
4	variables
5	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 23}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
6_kernel_parameter
_bias_parameter
7trainable_variables
8regularization_losses
9	variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?

_tf_keras_layer?
{"name": "layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}, "shared_object_id": 33}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 36, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 37}
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
8
9
10
11
12
13"
trackable_list_wrapper
?

?layers
@layer_regularization_losses
Anon_trainable_variables
trainable_variables
Bmetrics
regularization_losses
	variables
Clayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:?2layer_0/bias
*:(?2layer_0/igdn_0/reparam_beta
0:.
??2layer_0/igdn_0/reparam_gamma
':%
??2layer_0/kernel_rdft
:?2layer_1/bias
*:(?2layer_1/igdn_1/reparam_beta
0:.
??2layer_1/igdn_1/reparam_gamma
':%
??2layer_1/kernel_rdft
:?2layer_2/bias
*:(?2layer_2/igdn_2/reparam_beta
0:.
??2layer_2/igdn_2/reparam_gamma
':%
??2layer_2/kernel_rdft
:2layer_3/bias
&:$	?2layer_3/kernel_rdft
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?
D_beta_parameter
E_gamma_parameter
Ftrainable_variables
Gregularization_losses
H	variables
I	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "igdn_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?

Jlayers
Klayer_regularization_losses
Lnon_trainable_variables
Mmetrics
&trainable_variables
'regularization_losses
(	variables
Nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
O_beta_parameter
P_gamma_parameter
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "igdn_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?

Ulayers
Vlayer_regularization_losses
Wnon_trainable_variables
Xmetrics
,trainable_variables
-regularization_losses
.	variables
Ylayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Z_beta_parameter
[_gamma_parameter
\trainable_variables
]regularization_losses
^	variables
_	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "igdn_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?

`layers
alayer_regularization_losses
bnon_trainable_variables
cmetrics
2trainable_variables
3regularization_losses
4	variables
dlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
rdft"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

elayers
flayer_regularization_losses
gnon_trainable_variables
hmetrics
7trainable_variables
8regularization_losses
9	variables
ilayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

jlayers
klayer_regularization_losses
lnon_trainable_variables
mmetrics
;trainable_variables
<regularization_losses
=	variables
nlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
0
	1

2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
,
variable"
_generic_user_object
,
variable"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

olayers
player_regularization_losses
qnon_trainable_variables
rmetrics
Ftrainable_variables
Gregularization_losses
H	variables
slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
,
variable"
_generic_user_object
,
variable"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

tlayers
ulayer_regularization_losses
vnon_trainable_variables
wmetrics
Qtrainable_variables
Rregularization_losses
S	variables
xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
,
variable"
_generic_user_object
,
variable"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

ylayers
zlayer_regularization_losses
{non_trainable_variables
|metrics
\trainable_variables
]regularization_losses
^	variables
}layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
00"
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
?2?
A__inference_model_layer_call_and_return_conditional_losses_201626
A__inference_model_layer_call_and_return_conditional_losses_201703
A__inference_model_layer_call_and_return_conditional_losses_202615
A__inference_model_layer_call_and_return_conditional_losses_203139?
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
?2?
!__inference__wrapped_model_200517?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *A?>
<?9
input_2,????????????????????????????
?2?
&__inference_model_layer_call_fn_201858
&__inference_model_layer_call_fn_202012?
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
E__inference_synthesis_layer_call_and_return_conditional_losses_201144
E__inference_synthesis_layer_call_and_return_conditional_losses_201231
E__inference_synthesis_layer_call_and_return_conditional_losses_203663
E__inference_synthesis_layer_call_and_return_conditional_losses_204187?
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
?2?
*__inference_synthesis_layer_call_fn_201390
*__inference_synthesis_layer_call_fn_201548?
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
$__inference_signature_wrapper_202091input_2"?
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
 
?2?
C__inference_layer_0_layer_call_and_return_conditional_losses_204356?
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
?2??
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
C__inference_layer_1_layer_call_and_return_conditional_losses_204525?
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
?2??
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
C__inference_layer_2_layer_call_and_return_conditional_losses_204694?
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
?2??
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
C__inference_layer_3_layer_call_and_return_conditional_losses_204737?
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
?2??
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
?2?
D__inference_lambda_1_layer_call_and_return_conditional_losses_204743
D__inference_lambda_1_layer_call_and_return_conditional_losses_204749?
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec1
args)?&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults?

 
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
?2??
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
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7
J	
Const_8
J	
Const_9
J

Const_10
J

Const_11
J

Const_12
J

Const_13
J

Const_14
J

Const_15
J

Const_16
J

Const_17
J

Const_18
J

Const_19
J

Const_20
J

Const_21?
!__inference__wrapped_model_200517?:??????????????????????K?H
A?>
<?9
input_2,????????????????????????????
? "O?L
J
	synthesis=?:
	synthesis+????????????????????????????
D__inference_lambda_1_layer_call_and_return_conditional_losses_204743?Q?N
G?D
:?7
inputs+???????????????????????????

 
p
? "??<
5?2
0+???????????????????????????
? ?
D__inference_lambda_1_layer_call_and_return_conditional_losses_204749?Q?N
G?D
:?7
inputs+???????????????????????????

 
p 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_layer_0_layer_call_and_return_conditional_losses_204356????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_1_layer_call_and_return_conditional_losses_204525????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_2_layer_call_and_return_conditional_losses_204694????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_3_layer_call_and_return_conditional_losses_204737??J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_201626?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_201703?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_202615?:??????????????????????R?O
H?E
;?8
inputs,????????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_203139?:??????????????????????R?O
H?E
;?8
inputs,????????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
&__inference_model_layer_call_fn_201858?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p

 
? "2?/+????????????????????????????
&__inference_model_layer_call_fn_202012?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p 

 
? "2?/+????????????????????????????
$__inference_signature_wrapper_202091?:??????????????????????V?S
? 
L?I
G
input_2<?9
input_2,????????????????????????????"O?L
J
	synthesis=?:
	synthesis+????????????????????????????
E__inference_synthesis_layer_call_and_return_conditional_losses_201144?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_synthesis_layer_call_and_return_conditional_losses_201231?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_synthesis_layer_call_and_return_conditional_losses_203663?:??????????????????????R?O
H?E
;?8
inputs,????????????????????????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
E__inference_synthesis_layer_call_and_return_conditional_losses_204187?:??????????????????????R?O
H?E
;?8
inputs,????????????????????????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
*__inference_synthesis_layer_call_fn_201390?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p

 
? "2?/+????????????????????????????
*__inference_synthesis_layer_call_fn_201548?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p 

 
? "2?/+???????????????????????????