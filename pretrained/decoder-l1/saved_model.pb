פ0
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
regularization_losses
trainable_variables
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
regularization_losses
trainable_variables
	variables
	keras_api
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
layer_regularization_losses
regularization_losses
 non_trainable_variables
!layer_metrics

"layers
trainable_variables
#metrics
	variables
 
?
$_activation
%_kernel_parameter
_bias_parameter
&regularization_losses
'trainable_variables
(	variables
)	keras_api
?
*_activation
+_kernel_parameter
_bias_parameter
,regularization_losses
-trainable_variables
.	variables
/	keras_api
?
0_activation
1_kernel_parameter
_bias_parameter
2regularization_losses
3trainable_variables
4	variables
5	keras_api
~
6_kernel_parameter
_bias_parameter
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
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
?layer_regularization_losses
regularization_losses
@non_trainable_variables
Alayer_metrics

Blayers
trainable_variables
Cmetrics
	variables
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
 
 
 

0
1
 
}
D_beta_parameter
E_gamma_parameter
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api


rdft
 

0
1
2
3

0
1
2
3
?
Jlayer_regularization_losses
&regularization_losses
Knon_trainable_variables
Llayer_metrics

Mlayers
'trainable_variables
Nmetrics
(	variables
}
O_beta_parameter
P_gamma_parameter
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api


rdft
 

0
1
2
3

0
1
2
3
?
Ulayer_regularization_losses
,regularization_losses
Vnon_trainable_variables
Wlayer_metrics

Xlayers
-trainable_variables
Ymetrics
.	variables
}
Z_beta_parameter
[_gamma_parameter
\regularization_losses
]trainable_variables
^	variables
_	keras_api


rdft
 

0
1
2
3

0
1
2
3
?
`layer_regularization_losses
2regularization_losses
anon_trainable_variables
blayer_metrics

clayers
3trainable_variables
dmetrics
4	variables


rdft
 

0
1

0
1
?
elayer_regularization_losses
7regularization_losses
fnon_trainable_variables
glayer_metrics

hlayers
8trainable_variables
imetrics
9	variables
 
 
 
?
jlayer_regularization_losses
;regularization_losses
knon_trainable_variables
llayer_metrics

mlayers
<trainable_variables
nmetrics
=	variables
 
 
 
#
0
	1

2
3
4
 

variable

variable
 

0
1

0
1
?
olayer_regularization_losses
Fregularization_losses
pnon_trainable_variables
qlayer_metrics

rlayers
Gtrainable_variables
smetrics
H	variables
 
 
 

$0
 

variable

variable
 

0
1

0
1
?
tlayer_regularization_losses
Qregularization_losses
unon_trainable_variables
vlayer_metrics

wlayers
Rtrainable_variables
xmetrics
S	variables
 
 
 

*0
 

variable

variable
 

0
1

0
1
?
ylayer_regularization_losses
\regularization_losses
znon_trainable_variables
{layer_metrics

|layers
]trainable_variables
}metrics
^	variables
 
 
 
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
$__inference_signature_wrapper_202052
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
__inference__traced_save_204827
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
"__inference__traced_restore_204879??*
?
z
igdn_1_cond_2_false_204457%
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
igdn_1_cond_2_cond_false_204466*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_2_cond_true_2044652
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
?
?
#igdn_1_cond_1_cond_cond_true_204392*
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
?i
?
C__inference_layer_2_layer_call_and_return_conditional_losses_204655

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
igdn_2_cond_false_204532*
output_shapes
: **
then_branchR
igdn_2_cond_true_2045312
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
igdn_2_cond_1_false_204543*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_1_true_2045422
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
_gradient_op_typeCustomGradient-204588*.
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
_gradient_op_typeCustomGradient-204598*.
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
_gradient_op_typeCustomGradient-204612*$
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
igdn_2_cond_2_false_204626*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_2_true_2046252
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
?
?
/model_synthesis_layer_2_igdn_2_cond_true_2003173
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
?
?
1model_synthesis_layer_2_igdn_2_cond_2_true_200411Y
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
?
?
0synthesis_layer_0_igdn_0_cond_2_cond_true_202196N
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
?
{
 layer_2_igdn_2_cond_false_2034645
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
?
s
igdn_0_cond_1_false_200535
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
igdn_0_cond_1_cond_false_200544*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_1_cond_true_2005432
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
?
h
layer_0_igdn_0_cond_true_203665#
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
?
?
2model_synthesis_layer_0_igdn_0_cond_2_false_200090U
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
7model_synthesis_layer_0_igdn_0_cond_2_cond_false_200099*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_0_igdn_0_cond_2_cond_true_2000982,
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
?
?
"layer_0_igdn_0_cond_1_false_203153.
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
'layer_0_igdn_0_cond_1_cond_false_203162*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_1_cond_true_2031612
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
?
?
"layer_2_igdn_2_cond_1_false_203999.
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
'layer_2_igdn_2_cond_1_cond_false_204008*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_1_cond_true_2040072
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
?
z
igdn_1_cond_2_false_200807%
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
igdn_1_cond_2_cond_false_200816*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_2_cond_true_2008152
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
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_204710

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
?
?
'layer_1_igdn_1_cond_2_cond_false_2039309
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
?
?
igdn_2_cond_2_cond_true_201004*
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
?
?
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202648K
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
igdn_2_cond_2_cond_false_201005)
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
?*
?
C__inference_layer_3_layer_call_and_return_conditional_losses_204698

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
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_201189

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
?
?
+synthesis_layer_0_igdn_0_cond_1_true_202104F
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
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201434

inputs
layer_0_201356"
layer_0_201358:
??
layer_0_201360:	?
layer_0_201362"
layer_0_201364:
??
layer_0_201366
layer_0_201368
layer_0_201370:	?
layer_0_201372
layer_0_201374
layer_0_201376
layer_1_201379"
layer_1_201381:
??
layer_1_201383:	?
layer_1_201385"
layer_1_201387:
??
layer_1_201389
layer_1_201391
layer_1_201393:	?
layer_1_201395
layer_1_201397
layer_1_201399
layer_2_201402"
layer_2_201404:
??
layer_2_201406:	?
layer_2_201408"
layer_2_201410:
??
layer_2_201412
layer_2_201414
layer_2_201416:	?
layer_2_201418
layer_2_201420
layer_2_201422
layer_3_201425!
layer_3_201427:	?
layer_3_201429:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_201356layer_0_201358layer_0_201360layer_0_201362layer_0_201364layer_0_201366layer_0_201368layer_0_201370layer_0_201372layer_0_201374layer_0_201376*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_2006472!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201379layer_1_201381layer_1_201383layer_1_201385layer_1_201387layer_1_201389layer_1_201391layer_1_201393layer_1_201395layer_1_201397layer_1_201399*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_2008362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201402layer_2_201404layer_2_201406layer_2_201408layer_2_201410layer_2_201412layer_2_201414layer_2_201416layer_2_201418layer_2_201420layer_2_201422*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_2010252!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201425layer_3_201427layer_3_201429*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_2010882!
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011892
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
?
?
#igdn_0_cond_1_cond_cond_true_204223*
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
?
[
igdn_2_cond_false_200902%
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
?
?
*__inference_synthesis_layer_call_fn_201351
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2012762
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
?
?
"layer_1_igdn_1_cond_2_false_2033975
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
'layer_1_igdn_1_cond_2_cond_false_203406*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_2_cond_true_2034052
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
?	
?
7model_synthesis_layer_1_igdn_1_cond_2_cond_false_200260Y
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
?
?
,synthesis_layer_0_igdn_0_cond_1_false_202105B
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
1synthesis_layer_0_igdn_0_cond_1_cond_false_202114*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_1_cond_true_2021132&
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
?
?
&layer_0_igdn_0_cond_2_cond_true_203768:
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
?
?
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_202445N
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
?
?
&__inference_model_layer_call_fn_201819
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
A__inference_model_layer_call_and_return_conditional_losses_2017442
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
?
?
'layer_1_igdn_1_cond_2_cond_false_2034069
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
?
?
igdn_0_cond_1_cond_true_204213"
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
?
P
igdn_0_cond_true_200523
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
"layer_2_igdn_2_cond_2_false_2035585
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
'layer_2_igdn_2_cond_2_cond_false_203567*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_2_cond_true_2035662
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
?
?
'layer_1_igdn_1_cond_1_cond_false_2033233
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
,layer_1_igdn_1_cond_1_cond_cond_false_203333*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_1_igdn_1_cond_1_cond_cond_true_2033322!
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
?
?
,synthesis_layer_1_igdn_1_cond_2_false_202349I
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
1synthesis_layer_1_igdn_1_cond_2_cond_false_202358*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_2_cond_true_2023572&
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
?
?
"layer_0_igdn_0_cond_1_false_203677.
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
'layer_0_igdn_0_cond_1_cond_false_203686*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_1_cond_true_2036852
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
$igdn_2_cond_1_cond_cond_false_200932'
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
?
?
A__inference_model_layer_call_and_return_conditional_losses_201587
input_2
synthesis_201513$
synthesis_201515:
??
synthesis_201517:	?
synthesis_201519$
synthesis_201521:
??
synthesis_201523
synthesis_201525
synthesis_201527:	?
synthesis_201529
synthesis_201531
synthesis_201533
synthesis_201535$
synthesis_201537:
??
synthesis_201539:	?
synthesis_201541$
synthesis_201543:
??
synthesis_201545
synthesis_201547
synthesis_201549:	?
synthesis_201551
synthesis_201553
synthesis_201555
synthesis_201557$
synthesis_201559:
??
synthesis_201561:	?
synthesis_201563$
synthesis_201565:
??
synthesis_201567
synthesis_201569
synthesis_201571:	?
synthesis_201573
synthesis_201575
synthesis_201577
synthesis_201579#
synthesis_201581:	?
synthesis_201583:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinput_2synthesis_201513synthesis_201515synthesis_201517synthesis_201519synthesis_201521synthesis_201523synthesis_201525synthesis_201527synthesis_201529synthesis_201531synthesis_201533synthesis_201535synthesis_201537synthesis_201539synthesis_201541synthesis_201543synthesis_201545synthesis_201547synthesis_201549synthesis_201551synthesis_201553synthesis_201555synthesis_201557synthesis_201559synthesis_201561synthesis_201563synthesis_201565synthesis_201567synthesis_201569synthesis_201571synthesis_201573synthesis_201575synthesis_201577synthesis_201579synthesis_201581synthesis_201583*0
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2012762#
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
?
?
'layer_0_igdn_0_cond_1_cond_false_2036863
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
,layer_0_igdn_0_cond_1_cond_cond_false_203696*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_0_igdn_0_cond_1_cond_cond_true_2036952!
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
$igdn_1_cond_1_cond_cond_false_200743'
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
?
?
A__inference_model_layer_call_and_return_conditional_losses_201744

inputs
synthesis_201670$
synthesis_201672:
??
synthesis_201674:	?
synthesis_201676$
synthesis_201678:
??
synthesis_201680
synthesis_201682
synthesis_201684:	?
synthesis_201686
synthesis_201688
synthesis_201690
synthesis_201692$
synthesis_201694:
??
synthesis_201696:	?
synthesis_201698$
synthesis_201700:
??
synthesis_201702
synthesis_201704
synthesis_201706:	?
synthesis_201708
synthesis_201710
synthesis_201712
synthesis_201714$
synthesis_201716:
??
synthesis_201718:	?
synthesis_201720$
synthesis_201722:
??
synthesis_201724
synthesis_201726
synthesis_201728:	?
synthesis_201730
synthesis_201732
synthesis_201734
synthesis_201736#
synthesis_201738:	?
synthesis_201740:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinputssynthesis_201670synthesis_201672synthesis_201674synthesis_201676synthesis_201678synthesis_201680synthesis_201682synthesis_201684synthesis_201686synthesis_201688synthesis_201690synthesis_201692synthesis_201694synthesis_201696synthesis_201698synthesis_201700synthesis_201702synthesis_201704synthesis_201706synthesis_201708synthesis_201710synthesis_201712synthesis_201714synthesis_201716synthesis_201718synthesis_201720synthesis_201722synthesis_201724synthesis_201726synthesis_201728synthesis_201730synthesis_201732synthesis_201734synthesis_201736synthesis_201738synthesis_201740*0
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2012762#
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
?
?
+synthesis_layer_0_igdn_0_cond_2_true_202711M
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
??
?
A__inference_model_layer_call_and_return_conditional_losses_203100

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
*synthesis_layer_0_igdn_0_cond_false_202618*
output_shapes
: *<
then_branch-R+
)synthesis_layer_0_igdn_0_cond_true_2026172
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
,synthesis_layer_0_igdn_0_cond_1_false_202629*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_1_true_2026282!
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
_gradient_op_typeCustomGradient-202674*.
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
_gradient_op_typeCustomGradient-202684*.
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
_gradient_op_typeCustomGradient-202698*$
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
,synthesis_layer_0_igdn_0_cond_2_false_202712*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_2_true_2027112!
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
*synthesis_layer_1_igdn_1_cond_false_202779*
output_shapes
: *<
then_branch-R+
)synthesis_layer_1_igdn_1_cond_true_2027782
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
,synthesis_layer_1_igdn_1_cond_1_false_202790*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_1_true_2027892!
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
_gradient_op_typeCustomGradient-202835*.
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
_gradient_op_typeCustomGradient-202845*.
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
_gradient_op_typeCustomGradient-202859*$
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
,synthesis_layer_1_igdn_1_cond_2_false_202873*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_2_true_2028722!
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
*synthesis_layer_2_igdn_2_cond_false_202940*
output_shapes
: *<
then_branch-R+
)synthesis_layer_2_igdn_2_cond_true_2029392
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
,synthesis_layer_2_igdn_2_cond_1_false_202951*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_1_true_2029502!
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
_gradient_op_typeCustomGradient-202996*.
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
_gradient_op_typeCustomGradient-203006*.
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
_gradient_op_typeCustomGradient-203020*$
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
,synthesis_layer_2_igdn_2_cond_2_false_203034*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_2_true_2030332!
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
?
*synthesis_layer_2_igdn_2_cond_false_202940I
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
?
?
1model_synthesis_layer_0_igdn_0_cond_2_true_200089Y
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
?
?
*synthesis_layer_0_igdn_0_cond_false_202618I
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
?
?
,layer_1_igdn_1_cond_1_cond_cond_false_2033337
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
?
?
igdn_2_cond_2_true_200995)
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
?
?
0synthesis_layer_2_igdn_2_cond_1_cond_true_202435F
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
?
?
&layer_1_igdn_1_cond_1_cond_true_2038462
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
?
?
/model_synthesis_layer_1_igdn_1_cond_true_2001563
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
?
?
0model_synthesis_layer_1_igdn_1_cond_false_200157U
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
?	
?
;model_synthesis_layer_1_igdn_1_cond_1_cond_cond_true_200186Z
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
?
?
)synthesis_layer_2_igdn_2_cond_true_202415-
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
?
?
,layer_1_igdn_1_cond_1_cond_cond_false_2038577
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
?
?
'layer_2_igdn_2_cond_1_cond_false_2034843
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
,layer_2_igdn_2_cond_1_cond_cond_false_203494*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_2_igdn_2_cond_1_cond_cond_true_2034932!
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
?
?
#igdn_0_cond_1_cond_cond_true_200553*
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
?
?
'layer_2_igdn_2_cond_2_cond_false_2035679
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
1model_synthesis_layer_0_igdn_0_cond_1_true_200006R
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
?
?
0synthesis_layer_2_igdn_2_cond_2_cond_true_202518N
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
?
?
2model_synthesis_layer_1_igdn_1_cond_1_false_200168N
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
7model_synthesis_layer_1_igdn_1_cond_1_cond_false_200177*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_1_igdn_1_cond_1_cond_true_2001762,
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
?
?
igdn_1_cond_1_cond_false_200733#
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
$igdn_1_cond_1_cond_cond_false_200743*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_1_cond_1_cond_cond_true_2007422
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
#igdn_1_cond_1_cond_cond_true_200742*
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
?
?
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_202123N
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
?
?
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202285K
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
?
?
+layer_2_igdn_2_cond_1_cond_cond_true_204017:
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
?
?
0synthesis_layer_1_igdn_1_cond_2_cond_true_202881N
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
?
?
igdn_2_cond_2_cond_true_204634*
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
?
?
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202446K
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
1model_synthesis_layer_2_igdn_2_cond_1_true_200328R
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
?
?
+synthesis_layer_2_igdn_2_cond_2_true_203033M
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
?
P
igdn_1_cond_true_200712
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
?
?
2model_synthesis_layer_1_igdn_1_cond_2_false_200251U
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
7model_synthesis_layer_1_igdn_1_cond_2_cond_false_200260*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_1_igdn_1_cond_2_cond_true_2002592,
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
?
?
*synthesis_layer_1_igdn_1_cond_false_202779I
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
?
?
igdn_0_cond_1_cond_false_200544#
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
$igdn_0_cond_1_cond_cond_false_200554*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_0_cond_1_cond_cond_true_2005532
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
?
?
,synthesis_layer_1_igdn_1_cond_1_false_202266B
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
1synthesis_layer_1_igdn_1_cond_1_cond_false_202275*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_1_cond_true_2022742&
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
?
?
*synthesis_layer_2_igdn_2_cond_false_202416I
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
?
?
,synthesis_layer_1_igdn_1_cond_2_false_202873I
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
1synthesis_layer_1_igdn_1_cond_2_cond_false_202882*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_2_cond_true_2028812&
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
'layer_2_igdn_2_cond_1_cond_false_2040083
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
,layer_2_igdn_2_cond_1_cond_cond_false_204018*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_2_igdn_2_cond_1_cond_cond_true_2040172!
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
?
?
!layer_2_igdn_2_cond_2_true_2035579
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
?
?
igdn_1_cond_2_cond_false_200816)
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
?
?
igdn_2_cond_1_cond_false_204552#
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
$igdn_2_cond_1_cond_cond_false_204562*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_2_cond_1_cond_cond_true_2045612
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
?
?
+layer_1_igdn_1_cond_1_cond_cond_true_203332:
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
?
?
"layer_2_igdn_2_cond_2_false_2040825
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
'layer_2_igdn_2_cond_2_cond_false_204091*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_2_cond_true_2040902
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
?
?
1synthesis_layer_0_igdn_0_cond_2_cond_false_202721M
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
?
?
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_202808N
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
igdn_1_cond_2_cond_false_204466)
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
'layer_0_igdn_0_cond_2_cond_false_2037699
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
?
?
&__inference_model_layer_call_fn_201973
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
A__inference_model_layer_call_and_return_conditional_losses_2018982
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
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201105
layer_0_input
layer_0_200648"
layer_0_200650:
??
layer_0_200652:	?
layer_0_200654"
layer_0_200656:
??
layer_0_200658
layer_0_200660
layer_0_200662:	?
layer_0_200664
layer_0_200666
layer_0_200668
layer_1_200837"
layer_1_200839:
??
layer_1_200841:	?
layer_1_200843"
layer_1_200845:
??
layer_1_200847
layer_1_200849
layer_1_200851:	?
layer_1_200853
layer_1_200855
layer_1_200857
layer_2_201026"
layer_2_201028:
??
layer_2_201030:	?
layer_2_201032"
layer_2_201034:
??
layer_2_201036
layer_2_201038
layer_2_201040:	?
layer_2_201042
layer_2_201044
layer_2_201046
layer_3_201089!
layer_3_201091:	?
layer_3_201093:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputlayer_0_200648layer_0_200650layer_0_200652layer_0_200654layer_0_200656layer_0_200658layer_0_200660layer_0_200662layer_0_200664layer_0_200666layer_0_200668*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_2006472!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_200837layer_1_200839layer_1_200841layer_1_200843layer_1_200845layer_1_200847layer_1_200849layer_1_200851layer_1_200853layer_1_200855layer_1_200857*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_2008362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201026layer_2_201028layer_2_201030layer_2_201032layer_2_201034layer_2_201036layer_2_201038layer_2_201040layer_2_201042layer_2_201044layer_2_201046*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_2010252!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201089layer_3_201091layer_3_201093*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_2010882!
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011022
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
?
?
'layer_0_igdn_0_cond_2_cond_false_2032459
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
?
?
igdn_1_cond_1_cond_true_204382"
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
?
?
$igdn_1_cond_1_cond_cond_false_204393'
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
?
?
,synthesis_layer_1_igdn_1_cond_1_false_202790B
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
1synthesis_layer_1_igdn_1_cond_1_cond_false_202799*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_1_igdn_1_cond_1_cond_true_2027982&
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
?
?
$__inference_signature_wrapper_202052
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
!__inference__wrapped_model_2004782
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
A__inference_model_layer_call_and_return_conditional_losses_201898

inputs
synthesis_201824$
synthesis_201826:
??
synthesis_201828:	?
synthesis_201830$
synthesis_201832:
??
synthesis_201834
synthesis_201836
synthesis_201838:	?
synthesis_201840
synthesis_201842
synthesis_201844
synthesis_201846$
synthesis_201848:
??
synthesis_201850:	?
synthesis_201852$
synthesis_201854:
??
synthesis_201856
synthesis_201858
synthesis_201860:	?
synthesis_201862
synthesis_201864
synthesis_201866
synthesis_201868$
synthesis_201870:
??
synthesis_201872:	?
synthesis_201874$
synthesis_201876:
??
synthesis_201878
synthesis_201880
synthesis_201882:	?
synthesis_201884
synthesis_201886
synthesis_201888
synthesis_201890#
synthesis_201892:	?
synthesis_201894:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinputssynthesis_201824synthesis_201826synthesis_201828synthesis_201830synthesis_201832synthesis_201834synthesis_201836synthesis_201838synthesis_201840synthesis_201842synthesis_201844synthesis_201846synthesis_201848synthesis_201850synthesis_201852synthesis_201854synthesis_201856synthesis_201858synthesis_201860synthesis_201862synthesis_201864synthesis_201866synthesis_201868synthesis_201870synthesis_201872synthesis_201874synthesis_201876synthesis_201878synthesis_201880synthesis_201882synthesis_201884synthesis_201886synthesis_201888synthesis_201890synthesis_201892synthesis_201894*0
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2014342#
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
?*
?
C__inference_layer_3_layer_call_and_return_conditional_losses_201088

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
?
?
$igdn_0_cond_1_cond_cond_false_204224'
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
?
?
)synthesis_layer_2_igdn_2_cond_true_202939-
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
C__inference_layer_1_layer_call_and_return_conditional_losses_200836

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
igdn_1_cond_false_200713*
output_shapes
: **
then_branchR
igdn_1_cond_true_2007122
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
igdn_1_cond_1_false_200724*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_1_true_2007232
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
_gradient_op_typeCustomGradient-200769*.
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
_gradient_op_typeCustomGradient-200779*.
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
_gradient_op_typeCustomGradient-200793*$
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
igdn_1_cond_2_false_200807*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_2_true_2008062
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
z
igdn_0_cond_2_false_204288%
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
igdn_0_cond_2_cond_false_204297*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_2_cond_true_2042962
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
0synthesis_layer_0_igdn_0_cond_1_cond_true_202637F
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
1synthesis_layer_1_igdn_1_cond_2_cond_false_202882M
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
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202809K
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
*synthesis_layer_0_igdn_0_cond_false_202094I
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
?
?
'layer_0_igdn_0_cond_1_cond_false_2031623
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
,layer_0_igdn_0_cond_1_cond_cond_false_203172*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_0_igdn_0_cond_1_cond_cond_true_2031712!
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
?
?
,layer_0_igdn_0_cond_1_cond_cond_false_2031727
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
?
"layer_1_igdn_1_cond_1_false_203838.
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
'layer_1_igdn_1_cond_1_cond_false_203847*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_1_cond_true_2038462
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
?
?
igdn_0_cond_2_cond_false_204297)
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
s
igdn_1_cond_1_false_200724
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
igdn_1_cond_1_cond_false_200733*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_1_cond_true_2007322
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
?
?
!layer_2_igdn_2_cond_1_true_2034742
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
?	
?
<model_synthesis_layer_0_igdn_0_cond_1_cond_cond_false_200026W
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
?
?
6model_synthesis_layer_2_igdn_2_cond_1_cond_true_200337R
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
?
?
&layer_1_igdn_1_cond_1_cond_true_2033222
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
?
igdn_0_cond_1_cond_false_204214#
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
$igdn_0_cond_1_cond_cond_false_204224*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_0_cond_1_cond_cond_true_2042232
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
?
?
igdn_0_cond_2_cond_true_200626*
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
?
?
igdn_1_cond_2_true_204456)
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
?
?
"layer_0_igdn_0_cond_2_false_2037605
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
'layer_0_igdn_0_cond_2_cond_false_203769*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_2_cond_true_2037682
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
0synthesis_layer_0_igdn_0_cond_1_cond_true_202113F
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
?
?
'layer_2_igdn_2_cond_2_cond_false_2040919
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
?
{
 layer_1_igdn_1_cond_false_2033035
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
?	
?
;model_synthesis_layer_0_igdn_0_cond_1_cond_cond_true_200025Z
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
?
?
&layer_2_igdn_2_cond_2_cond_true_204090:
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
?
?
+layer_2_igdn_2_cond_1_cond_cond_true_203493:
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
<model_synthesis_layer_1_igdn_1_cond_1_cond_cond_false_200187W
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
?
?
/model_synthesis_layer_0_igdn_0_cond_true_1999953
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
?
?
+synthesis_layer_1_igdn_1_cond_1_true_202789F
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
?
z
igdn_0_cond_1_true_204204"
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
?
z
igdn_2_cond_1_true_200912"
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
?
z
igdn_0_cond_1_true_200534"
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
?
?
!layer_2_igdn_2_cond_2_true_2040819
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
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201192
layer_0_input
layer_0_201108"
layer_0_201110:
??
layer_0_201112:	?
layer_0_201114"
layer_0_201116:
??
layer_0_201118
layer_0_201120
layer_0_201122:	?
layer_0_201124
layer_0_201126
layer_0_201128
layer_1_201131"
layer_1_201133:
??
layer_1_201135:	?
layer_1_201137"
layer_1_201139:
??
layer_1_201141
layer_1_201143
layer_1_201145:	?
layer_1_201147
layer_1_201149
layer_1_201151
layer_2_201154"
layer_2_201156:
??
layer_2_201158:	?
layer_2_201160"
layer_2_201162:
??
layer_2_201164
layer_2_201166
layer_2_201168:	?
layer_2_201170
layer_2_201172
layer_2_201174
layer_3_201177!
layer_3_201179:	?
layer_3_201181:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllayer_0_inputlayer_0_201108layer_0_201110layer_0_201112layer_0_201114layer_0_201116layer_0_201118layer_0_201120layer_0_201122layer_0_201124layer_0_201126layer_0_201128*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_2006472!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201131layer_1_201133layer_1_201135layer_1_201137layer_1_201139layer_1_201141layer_1_201143layer_1_201145layer_1_201147layer_1_201149layer_1_201151*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_2008362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201154layer_2_201156layer_2_201158layer_2_201160layer_2_201162layer_2_201164layer_2_201166layer_2_201168layer_2_201170layer_2_201172layer_2_201174*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_2010252!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201177layer_3_201179layer_3_201181*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_2010882!
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011892
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
?
?
0synthesis_layer_1_igdn_1_cond_1_cond_true_202274F
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
+synthesis_layer_2_igdn_2_cond_1_true_202426F
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
??
?
E__inference_synthesis_layer_call_and_return_conditional_losses_203624

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
 layer_0_igdn_0_cond_false_203142*
output_shapes
: *2
then_branch#R!
layer_0_igdn_0_cond_true_2031412
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
"layer_0_igdn_0_cond_1_false_203153*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_1_true_2031522
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
_gradient_op_typeCustomGradient-203198*.
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
_gradient_op_typeCustomGradient-203208*.
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
_gradient_op_typeCustomGradient-203222*$
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
"layer_0_igdn_0_cond_2_false_203236*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_2_true_2032352
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
 layer_1_igdn_1_cond_false_203303*
output_shapes
: *2
then_branch#R!
layer_1_igdn_1_cond_true_2033022
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
"layer_1_igdn_1_cond_1_false_203314*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_1_true_2033132
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
_gradient_op_typeCustomGradient-203359*.
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
_gradient_op_typeCustomGradient-203369*.
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
_gradient_op_typeCustomGradient-203383*$
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
"layer_1_igdn_1_cond_2_false_203397*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_2_true_2033962
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
 layer_2_igdn_2_cond_false_203464*
output_shapes
: *2
then_branch#R!
layer_2_igdn_2_cond_true_2034632
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
"layer_2_igdn_2_cond_1_false_203475*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_1_true_2034742
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
_gradient_op_typeCustomGradient-203520*.
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
_gradient_op_typeCustomGradient-203530*.
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
_gradient_op_typeCustomGradient-203544*$
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
"layer_2_igdn_2_cond_2_false_203558*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_2_true_2035572
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
?
?
2model_synthesis_layer_0_igdn_0_cond_1_false_200007N
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
7model_synthesis_layer_0_igdn_0_cond_1_cond_false_200016*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_0_igdn_0_cond_1_cond_true_2000152,
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
?
?
1synthesis_layer_2_igdn_2_cond_2_cond_false_202519M
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
?
?
igdn_2_cond_1_cond_true_200921"
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
?
?
0synthesis_layer_2_igdn_2_cond_1_cond_true_202959F
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
?
?
"layer_1_igdn_1_cond_2_false_2039215
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
'layer_1_igdn_1_cond_2_cond_false_203930*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_2_cond_true_2039292
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
?
?
)synthesis_layer_1_igdn_1_cond_true_202254-
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
?
?
!layer_1_igdn_1_cond_2_true_2039209
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
?
s
igdn_2_cond_1_false_204543
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
igdn_2_cond_1_cond_false_204552*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_1_cond_true_2045512
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
?
[
igdn_2_cond_false_204532%
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
?
[
igdn_0_cond_false_200524%
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
?
P
igdn_2_cond_true_200901
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
?)
?
__inference__traced_save_204827
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
?
?
"layer_1_igdn_1_cond_1_false_203314.
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
'layer_1_igdn_1_cond_1_cond_false_203323*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_1_igdn_1_cond_1_cond_true_2033222
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
?
?
igdn_2_cond_2_cond_false_204635)
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
?
?
+synthesis_layer_2_igdn_2_cond_1_true_202950F
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
?i
?
C__inference_layer_0_layer_call_and_return_conditional_losses_204317

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
igdn_0_cond_false_204194*
output_shapes
: **
then_branchR
igdn_0_cond_true_2041932
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
igdn_0_cond_1_false_204205*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_1_true_2042042
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
_gradient_op_typeCustomGradient-204250*.
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
_gradient_op_typeCustomGradient-204260*.
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
_gradient_op_typeCustomGradient-204274*$
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
igdn_0_cond_2_false_204288*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_2_true_2042872
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
?
s
igdn_1_cond_1_false_204374
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
igdn_1_cond_1_cond_false_204383*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_1_cond_1_cond_true_2043822
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
?	
?
7model_synthesis_layer_2_igdn_2_cond_2_cond_false_200421Y
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
+synthesis_layer_1_igdn_1_cond_2_true_202348M
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
?
{
 layer_0_igdn_0_cond_false_2031425
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
?
?
2model_synthesis_layer_2_igdn_2_cond_2_false_200412U
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
7model_synthesis_layer_2_igdn_2_cond_2_cond_false_200421*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_2_igdn_2_cond_2_cond_true_2004202,
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
?i
?
C__inference_layer_2_layer_call_and_return_conditional_losses_201025

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
igdn_2_cond_false_200902*
output_shapes
: **
then_branchR
igdn_2_cond_true_2009012
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
igdn_2_cond_1_false_200913*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_1_true_2009122
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
_gradient_op_typeCustomGradient-200958*.
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
_gradient_op_typeCustomGradient-200968*.
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
_gradient_op_typeCustomGradient-200982*$
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
igdn_2_cond_2_false_200996*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_2_cond_2_true_2009952
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
?
?
6model_synthesis_layer_1_igdn_1_cond_1_cond_true_200176R
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
?
?
igdn_1_cond_2_cond_true_200815*
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
?
?
,synthesis_layer_0_igdn_0_cond_2_false_202188I
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
1synthesis_layer_0_igdn_0_cond_2_cond_false_202197*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_2_cond_true_2021962&
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
?
[
igdn_0_cond_false_204194%
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
??
?	
"__inference__traced_restore_204879
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
?
?
&layer_1_igdn_1_cond_2_cond_true_203405:
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
&layer_0_igdn_0_cond_1_cond_true_2036852
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
?
P
igdn_2_cond_true_204531
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
?
[
igdn_1_cond_false_204363%
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
?
?
&layer_0_igdn_0_cond_2_cond_true_203244:
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
?
?
igdn_2_cond_2_true_204625)
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
?
[
igdn_1_cond_false_200713%
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
?
?
,layer_0_igdn_0_cond_1_cond_cond_false_2036967
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
?
{
 layer_1_igdn_1_cond_false_2038275
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
?
?
6model_synthesis_layer_1_igdn_1_cond_2_cond_true_200259Z
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
?
?
0model_synthesis_layer_2_igdn_2_cond_false_200318U
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
?
?
+layer_0_igdn_0_cond_1_cond_cond_true_203695:
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
?
?
!layer_0_igdn_0_cond_2_true_2037599
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
?
?
'layer_1_igdn_1_cond_1_cond_false_2038473
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
,layer_1_igdn_1_cond_1_cond_cond_false_203857*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+layer_1_igdn_1_cond_1_cond_cond_true_2038562!
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
s
igdn_0_cond_1_false_204205
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
igdn_0_cond_1_cond_false_204214*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_1_cond_true_2042132
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
?
?
2model_synthesis_layer_2_igdn_2_cond_1_false_200329N
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
7model_synthesis_layer_2_igdn_2_cond_1_cond_false_200338*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6model_synthesis_layer_2_igdn_2_cond_1_cond_true_2003372,
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
1model_synthesis_layer_1_igdn_1_cond_2_true_200250Y
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
?
?
1synthesis_layer_2_igdn_2_cond_2_cond_false_203043M
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
?
z
igdn_1_cond_1_true_204373"
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
?
?
)synthesis_layer_0_igdn_0_cond_true_202093-
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
?
?
!layer_0_igdn_0_cond_1_true_2031522
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
?
?
+synthesis_layer_0_igdn_0_cond_2_true_202187M
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
?
?
+synthesis_layer_1_igdn_1_cond_1_true_202265F
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
?
?
igdn_2_cond_1_cond_true_204551"
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
?
z
igdn_2_cond_2_false_200996%
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
igdn_2_cond_2_cond_false_201005*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_2_cond_true_2010042
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
?
P
igdn_1_cond_true_204362
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
?
?
&layer_0_igdn_0_cond_1_cond_true_2031612
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
?
?
A__inference_model_layer_call_and_return_conditional_losses_201664
input_2
synthesis_201590$
synthesis_201592:
??
synthesis_201594:	?
synthesis_201596$
synthesis_201598:
??
synthesis_201600
synthesis_201602
synthesis_201604:	?
synthesis_201606
synthesis_201608
synthesis_201610
synthesis_201612$
synthesis_201614:
??
synthesis_201616:	?
synthesis_201618$
synthesis_201620:
??
synthesis_201622
synthesis_201624
synthesis_201626:	?
synthesis_201628
synthesis_201630
synthesis_201632
synthesis_201634$
synthesis_201636:
??
synthesis_201638:	?
synthesis_201640$
synthesis_201642:
??
synthesis_201644
synthesis_201646
synthesis_201648:	?
synthesis_201650
synthesis_201652
synthesis_201654
synthesis_201656#
synthesis_201658:	?
synthesis_201660:
identity??!synthesis/StatefulPartitionedCall?
!synthesis/StatefulPartitionedCallStatefulPartitionedCallinput_2synthesis_201590synthesis_201592synthesis_201594synthesis_201596synthesis_201598synthesis_201600synthesis_201602synthesis_201604synthesis_201606synthesis_201608synthesis_201610synthesis_201612synthesis_201614synthesis_201616synthesis_201618synthesis_201620synthesis_201622synthesis_201624synthesis_201626synthesis_201628synthesis_201630synthesis_201632synthesis_201634synthesis_201636synthesis_201638synthesis_201640synthesis_201642synthesis_201644synthesis_201646synthesis_201648synthesis_201650synthesis_201652synthesis_201654synthesis_201656synthesis_201658synthesis_201660*0
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2014342#
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
?
?
,layer_2_igdn_2_cond_1_cond_cond_false_2040187
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
?
?
igdn_0_cond_2_true_204287)
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
?
?
,synthesis_layer_0_igdn_0_cond_2_false_202712I
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
1synthesis_layer_0_igdn_0_cond_2_cond_false_202721*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_2_cond_true_2027202&
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
?
?
1synthesis_layer_1_igdn_1_cond_2_cond_false_202358M
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
?
?
1synthesis_layer_1_igdn_1_cond_1_cond_false_202275G
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
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202285*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_2022842+
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
?
!layer_0_igdn_0_cond_2_true_2032359
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
?i
?
C__inference_layer_0_layer_call_and_return_conditional_losses_200647

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
igdn_0_cond_false_200524*
output_shapes
: **
then_branchR
igdn_0_cond_true_2005232
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
igdn_0_cond_1_false_200535*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_1_true_2005342
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
_gradient_op_typeCustomGradient-200580*.
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
_gradient_op_typeCustomGradient-200590*.
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
_gradient_op_typeCustomGradient-200604*$
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
igdn_0_cond_2_false_200618*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_0_cond_2_true_2006172
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
?
h
layer_2_igdn_2_cond_true_203463#
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
?
?
igdn_1_cond_1_cond_false_204383#
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
$igdn_1_cond_1_cond_cond_false_204393*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_1_cond_1_cond_cond_true_2043922
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
?
?
"layer_2_igdn_2_cond_1_false_203475.
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
'layer_2_igdn_2_cond_1_cond_false_203484*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_2_igdn_2_cond_1_cond_true_2034832
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
?
?
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_202647N
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
?
?
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_202969N
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
?
7model_synthesis_layer_0_igdn_0_cond_2_cond_false_200099Y
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
?
?
,layer_2_igdn_2_cond_1_cond_cond_false_2034947
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
?
?
,synthesis_layer_2_igdn_2_cond_1_false_202427B
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
1synthesis_layer_2_igdn_2_cond_1_cond_false_202436*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_1_cond_true_2024352&
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
?
{
 layer_0_igdn_0_cond_false_2036665
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
?
?
$igdn_2_cond_1_cond_cond_false_204562'
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
?
?
+layer_1_igdn_1_cond_1_cond_cond_true_203856:
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
?
s
igdn_2_cond_1_false_200913
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
igdn_2_cond_1_cond_false_200922*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_1_cond_true_2009212
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
?
?
,synthesis_layer_0_igdn_0_cond_1_false_202629B
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
1synthesis_layer_0_igdn_0_cond_1_cond_false_202638*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_0_igdn_0_cond_1_cond_true_2026372&
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
?
?
7model_synthesis_layer_0_igdn_0_cond_1_cond_false_200016S
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
<model_synthesis_layer_0_igdn_0_cond_1_cond_cond_false_200026*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_0_igdn_0_cond_1_cond_cond_true_20002521
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
?
?
1synthesis_layer_0_igdn_0_cond_1_cond_false_202638G
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
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202648*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_2026472+
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
?
?
1synthesis_layer_2_igdn_2_cond_1_cond_false_202436G
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
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202446*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_2024452+
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
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_201102

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
)synthesis_layer_1_igdn_1_cond_true_202778-
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
?
?
0model_synthesis_layer_0_igdn_0_cond_false_199996U
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
?
?
"layer_0_igdn_0_cond_2_false_2032365
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
'layer_0_igdn_0_cond_2_cond_false_203245*A
output_shapes0
.:,????????????????????????????*9
then_branch*R(
&layer_0_igdn_0_cond_2_cond_true_2032442
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
?
?
!layer_0_igdn_0_cond_1_true_2036762
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
?
?
&layer_2_igdn_2_cond_1_cond_true_2040072
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
1synthesis_layer_0_igdn_0_cond_2_cond_false_202197M
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
?
z
igdn_1_cond_1_true_200723"
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
+synthesis_layer_2_igdn_2_cond_2_true_202509M
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
&layer_1_igdn_1_cond_2_cond_true_203929:
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
?
?
6model_synthesis_layer_0_igdn_0_cond_1_cond_true_200015R
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
??
?
E__inference_synthesis_layer_call_and_return_conditional_losses_204148

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
 layer_0_igdn_0_cond_false_203666*
output_shapes
: *2
then_branch#R!
layer_0_igdn_0_cond_true_2036652
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
"layer_0_igdn_0_cond_1_false_203677*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_1_true_2036762
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
_gradient_op_typeCustomGradient-203722*.
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
_gradient_op_typeCustomGradient-203732*.
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
_gradient_op_typeCustomGradient-203746*$
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
"layer_0_igdn_0_cond_2_false_203760*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_0_igdn_0_cond_2_true_2037592
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
 layer_1_igdn_1_cond_false_203827*
output_shapes
: *2
then_branch#R!
layer_1_igdn_1_cond_true_2038262
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
"layer_1_igdn_1_cond_1_false_203838*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_1_true_2038372
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
_gradient_op_typeCustomGradient-203883*.
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
_gradient_op_typeCustomGradient-203893*.
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
_gradient_op_typeCustomGradient-203907*$
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
"layer_1_igdn_1_cond_2_false_203921*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_1_igdn_1_cond_2_true_2039202
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
 layer_2_igdn_2_cond_false_203988*
output_shapes
: *2
then_branch#R!
layer_2_igdn_2_cond_true_2039872
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
"layer_2_igdn_2_cond_1_false_203999*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_1_true_2039982
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
_gradient_op_typeCustomGradient-204044*.
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
_gradient_op_typeCustomGradient-204054*.
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
_gradient_op_typeCustomGradient-204068*$
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
"layer_2_igdn_2_cond_2_false_204082*A
output_shapes0
.:,????????????????????????????*4
then_branch%R#
!layer_2_igdn_2_cond_2_true_2040812
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
?	
?
;model_synthesis_layer_2_igdn_2_cond_1_cond_cond_true_200347Z
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
?
?
igdn_1_cond_2_true_200806)
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
?
h
layer_0_igdn_0_cond_true_203141#
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
?
?
&layer_2_igdn_2_cond_1_cond_true_2034832
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
?
<model_synthesis_layer_2_igdn_2_cond_1_cond_cond_false_200348W
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
?*
?
E__inference_synthesis_layer_call_and_return_conditional_losses_201276

inputs
layer_0_201198"
layer_0_201200:
??
layer_0_201202:	?
layer_0_201204"
layer_0_201206:
??
layer_0_201208
layer_0_201210
layer_0_201212:	?
layer_0_201214
layer_0_201216
layer_0_201218
layer_1_201221"
layer_1_201223:
??
layer_1_201225:	?
layer_1_201227"
layer_1_201229:
??
layer_1_201231
layer_1_201233
layer_1_201235:	?
layer_1_201237
layer_1_201239
layer_1_201241
layer_2_201244"
layer_2_201246:
??
layer_2_201248:	?
layer_2_201250"
layer_2_201252:
??
layer_2_201254
layer_2_201256
layer_2_201258:	?
layer_2_201260
layer_2_201262
layer_2_201264
layer_3_201267!
layer_3_201269:	?
layer_3_201271:
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCallinputslayer_0_201198layer_0_201200layer_0_201202layer_0_201204layer_0_201206layer_0_201208layer_0_201210layer_0_201212layer_0_201214layer_0_201216layer_0_201218*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_2006472!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_201221layer_1_201223layer_1_201225layer_1_201227layer_1_201229layer_1_201231layer_1_201233layer_1_201235layer_1_201237layer_1_201239layer_1_201241*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_2008362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_201244layer_2_201246layer_2_201248layer_2_201250layer_2_201252layer_2_201254layer_2_201256layer_2_201258layer_2_201260layer_2_201262layer_2_201264*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_2010252!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_201267layer_3_201269layer_3_201271*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_2010882!
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_2011022
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
?
?
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_202284N
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
?
?
&layer_2_igdn_2_cond_2_cond_true_203566:
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
)synthesis_layer_0_igdn_0_cond_true_202617-
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
?
z
igdn_2_cond_1_true_204542"
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
?
?
+synthesis_layer_1_igdn_1_cond_2_true_202872M
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
igdn_0_cond_2_cond_true_204296*
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
?
?
igdn_0_cond_1_cond_true_200543"
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
?i
?
C__inference_layer_1_layer_call_and_return_conditional_losses_204486

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
igdn_1_cond_false_204363*
output_shapes
: **
then_branchR
igdn_1_cond_true_2043622
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
igdn_1_cond_1_false_204374*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_1_true_2043732
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
_gradient_op_typeCustomGradient-204419*.
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
_gradient_op_typeCustomGradient-204429*.
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
_gradient_op_typeCustomGradient-204443*$
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
igdn_1_cond_2_false_204457*A
output_shapes0
.:,????????????????????????????*,
then_branchR
igdn_1_cond_2_true_2044562
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
?
?
1synthesis_layer_1_igdn_1_cond_1_cond_false_202799G
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
6synthesis_layer_1_igdn_1_cond_1_cond_cond_false_202809*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_1_igdn_1_cond_1_cond_cond_true_2028082+
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
?
z
igdn_0_cond_2_false_200618%
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
igdn_0_cond_2_cond_false_200627*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_0_cond_2_cond_true_2006262
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
?
?
!layer_1_igdn_1_cond_2_true_2033969
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
h
layer_1_igdn_1_cond_true_203302#
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
??
?
A__inference_model_layer_call_and_return_conditional_losses_202576

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
*synthesis_layer_0_igdn_0_cond_false_202094*
output_shapes
: *<
then_branch-R+
)synthesis_layer_0_igdn_0_cond_true_2020932
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
,synthesis_layer_0_igdn_0_cond_1_false_202105*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_1_true_2021042!
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
_gradient_op_typeCustomGradient-202150*.
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
_gradient_op_typeCustomGradient-202160*.
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
_gradient_op_typeCustomGradient-202174*$
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
,synthesis_layer_0_igdn_0_cond_2_false_202188*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_0_igdn_0_cond_2_true_2021872!
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
*synthesis_layer_1_igdn_1_cond_false_202255*
output_shapes
: *<
then_branch-R+
)synthesis_layer_1_igdn_1_cond_true_2022542
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
,synthesis_layer_1_igdn_1_cond_1_false_202266*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_1_true_2022652!
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
_gradient_op_typeCustomGradient-202311*.
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
_gradient_op_typeCustomGradient-202321*.
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
_gradient_op_typeCustomGradient-202335*$
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
,synthesis_layer_1_igdn_1_cond_2_false_202349*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_1_igdn_1_cond_2_true_2023482!
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
*synthesis_layer_2_igdn_2_cond_false_202416*
output_shapes
: *<
then_branch-R+
)synthesis_layer_2_igdn_2_cond_true_2024152
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
,synthesis_layer_2_igdn_2_cond_1_false_202427*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_1_true_2024262!
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
_gradient_op_typeCustomGradient-202472*.
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
_gradient_op_typeCustomGradient-202482*.
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
_gradient_op_typeCustomGradient-202496*$
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
,synthesis_layer_2_igdn_2_cond_2_false_202510*A
output_shapes0
.:,????????????????????????????*>
then_branch/R-
+synthesis_layer_2_igdn_2_cond_2_true_2025092!
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
0synthesis_layer_1_igdn_1_cond_2_cond_true_202357N
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
igdn_0_cond_true_204193
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
?
?
,synthesis_layer_2_igdn_2_cond_2_false_203034I
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
1synthesis_layer_2_igdn_2_cond_2_cond_false_203043*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_2_cond_true_2030422&
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
?
?
igdn_1_cond_2_cond_true_204465*
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
!layer_1_igdn_1_cond_1_true_2033132
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
#igdn_2_cond_1_cond_cond_true_200931*
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
?
?
7model_synthesis_layer_2_igdn_2_cond_1_cond_false_200338S
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
<model_synthesis_layer_2_igdn_2_cond_1_cond_cond_false_200348*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_2_igdn_2_cond_1_cond_cond_true_20034721
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
?
?
*synthesis_layer_1_igdn_1_cond_false_202255I
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
?
?
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202124K
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
1model_synthesis_layer_1_igdn_1_cond_1_true_200167R
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
?
?
,synthesis_layer_2_igdn_2_cond_2_false_202510I
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
1synthesis_layer_2_igdn_2_cond_2_cond_false_202519*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_2_cond_true_2025182&
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
?
?
#igdn_2_cond_1_cond_cond_true_204561*
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
??
?
!__inference__wrapped_model_200478
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
0model_synthesis_layer_0_igdn_0_cond_false_199996*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_0_igdn_0_cond_true_1999952%
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
2model_synthesis_layer_0_igdn_0_cond_1_false_200007*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_0_igdn_0_cond_1_true_2000062'
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
_gradient_op_typeCustomGradient-200052*.
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
_gradient_op_typeCustomGradient-200062*.
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
_gradient_op_typeCustomGradient-200076*$
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
2model_synthesis_layer_0_igdn_0_cond_2_false_200090*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_0_igdn_0_cond_2_true_2000892'
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
0model_synthesis_layer_1_igdn_1_cond_false_200157*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_1_igdn_1_cond_true_2001562%
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
2model_synthesis_layer_1_igdn_1_cond_1_false_200168*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_1_igdn_1_cond_1_true_2001672'
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
_gradient_op_typeCustomGradient-200213*.
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
_gradient_op_typeCustomGradient-200223*.
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
_gradient_op_typeCustomGradient-200237*$
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
2model_synthesis_layer_1_igdn_1_cond_2_false_200251*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_1_igdn_1_cond_2_true_2002502'
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
0model_synthesis_layer_2_igdn_2_cond_false_200318*
output_shapes
: *B
then_branch3R1
/model_synthesis_layer_2_igdn_2_cond_true_2003172%
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
2model_synthesis_layer_2_igdn_2_cond_1_false_200329*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_2_igdn_2_cond_1_true_2003282'
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
_gradient_op_typeCustomGradient-200374*.
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
_gradient_op_typeCustomGradient-200384*.
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
_gradient_op_typeCustomGradient-200398*$
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
2model_synthesis_layer_2_igdn_2_cond_2_false_200412*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1model_synthesis_layer_2_igdn_2_cond_2_true_2004112'
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
?
?
igdn_1_cond_1_cond_true_200732"
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
?
h
layer_2_igdn_2_cond_true_203987#
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
?
?
+layer_0_igdn_0_cond_1_cond_cond_true_203171:
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
?
?
0synthesis_layer_1_igdn_1_cond_1_cond_true_202798F
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
?
?
6model_synthesis_layer_0_igdn_0_cond_2_cond_true_200098Z
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
?
?
igdn_0_cond_2_true_200617)
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
?
z
igdn_2_cond_2_false_204626%
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
igdn_2_cond_2_cond_false_204635*A
output_shapes0
.:,????????????????????????????*1
then_branch"R 
igdn_2_cond_2_cond_true_2046342
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
?
?
*__inference_synthesis_layer_call_fn_201509
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
E__inference_synthesis_layer_call_and_return_conditional_losses_2014342
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
?
`
D__inference_lambda_1_layer_call_and_return_conditional_losses_204704

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
?
?
,synthesis_layer_2_igdn_2_cond_1_false_202951B
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
1synthesis_layer_2_igdn_2_cond_1_cond_false_202960*A
output_shapes0
.:,????????????????????????????*C
then_branch4R2
0synthesis_layer_2_igdn_2_cond_1_cond_true_2029592&
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
+synthesis_layer_0_igdn_0_cond_1_true_202628F
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
?
?
!layer_1_igdn_1_cond_1_true_2038372
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
?
?
7model_synthesis_layer_1_igdn_1_cond_1_cond_false_200177S
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
<model_synthesis_layer_1_igdn_1_cond_1_cond_cond_false_200187*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;model_synthesis_layer_1_igdn_1_cond_1_cond_cond_true_20018621
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
?
{
 layer_2_igdn_2_cond_false_2039885
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
?
?
igdn_2_cond_1_cond_false_200922#
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
$igdn_2_cond_1_cond_cond_false_200932*A
output_shapes0
.:,????????????????????????????*6
then_branch'R%
#igdn_2_cond_1_cond_cond_true_2009312
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
?
?
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202970K
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
?
?
6model_synthesis_layer_2_igdn_2_cond_2_cond_true_200420Z
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
1synthesis_layer_0_igdn_0_cond_1_cond_false_202114G
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
6synthesis_layer_0_igdn_0_cond_1_cond_cond_false_202124*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_0_igdn_0_cond_1_cond_cond_true_2021232+
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
?
?
1synthesis_layer_2_igdn_2_cond_1_cond_false_202960G
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
6synthesis_layer_2_igdn_2_cond_1_cond_cond_false_202970*A
output_shapes0
.:,????????????????????????????*H
then_branch9R7
5synthesis_layer_2_igdn_2_cond_1_cond_cond_true_2029692+
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
?
?
0synthesis_layer_0_igdn_0_cond_2_cond_true_202720N
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
?
?
0synthesis_layer_2_igdn_2_cond_2_cond_true_203042N
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
?
h
layer_1_igdn_1_cond_true_203826#
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
igdn_0_cond_2_cond_false_200627)
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
?
?
!layer_2_igdn_2_cond_1_true_2039982
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
?
?
$igdn_0_cond_1_cond_cond_false_200554'
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
regularization_losses
trainable_variables
	variables
	keras_api

signatures
~_default_save_signature
__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "name": "synthesis", "inbound_nodes": [[["input_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["synthesis", 1, 0]]}, "shared_object_id": 39, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 192]}, "float32", "input_2"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "name": "synthesis", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 38}], "input_layers": [["input_2", 0, 0]], "output_layers": [["synthesis", 1, 0]]}}}
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
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequential??{"name": "synthesis", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}]}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 192]}, "float32", "layer_0_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "SynthesisTransform", "config": {"name": "synthesis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 192]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "layer_0_input"}, "shared_object_id": 1}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 2}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 12}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 13}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 22}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 23}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 32}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}, "shared_object_id": 33}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 36}, {"class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 37}]}}}
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
layer_regularization_losses
regularization_losses
 non_trainable_variables
!layer_metrics

"layers
trainable_variables
#metrics
	variables
__call__
~_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
$_activation
%_kernel_parameter
_bias_parameter
&regularization_losses
'trainable_variables
(	variables
)	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 2}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 12, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
*_activation
+_kernel_parameter
_bias_parameter
,regularization_losses
-trainable_variables
.	variables
/	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 13}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 22, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
0_activation
1_kernel_parameter
_bias_parameter
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 23}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 32, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
6_kernel_parameter
_bias_parameter
7regularization_losses
8trainable_variables
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": false, "strides_down": {"class_name": "__tuple__", "items": [1, 1]}, "strides_up": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 3]}, "dtype": "float32"}, "shared_object_id": 33}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 36, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "lambda_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda_1", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBFABTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+YgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 37}
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
?layer_regularization_losses
regularization_losses
@non_trainable_variables
Alayer_metrics

Blayers
trainable_variables
Cmetrics
	variables
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
D_beta_parameter
E_gamma_parameter
Fregularization_losses
Gtrainable_variables
H	variables
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "igdn_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_0", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 3}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 4}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 5}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 8}}, "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
Jlayer_regularization_losses
&regularization_losses
Knon_trainable_variables
Llayer_metrics

Mlayers
'trainable_variables
Nmetrics
(	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
O_beta_parameter
P_gamma_parameter
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "igdn_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_1", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 14}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 15}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 16}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}}, "shared_object_id": 19, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
Ulayer_regularization_losses
,regularization_losses
Vnon_trainable_variables
Wlayer_metrics

Xlayers
-trainable_variables
Ymetrics
.	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
Z_beta_parameter
[_gamma_parameter
\regularization_losses
]trainable_variables
^	variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "igdn_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "igdn_2", "trainable": true, "dtype": "float32", "inverse": true, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 24}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 25}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 26}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 7}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}}, "shared_object_id": 29, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
(
rdft"
_generic_user_object
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
`layer_regularization_losses
2regularization_losses
anon_trainable_variables
blayer_metrics

clayers
3trainable_variables
dmetrics
4	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
(
rdft"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
elayer_regularization_losses
7regularization_losses
fnon_trainable_variables
glayer_metrics

hlayers
8trainable_variables
imetrics
9	variables
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
jlayer_regularization_losses
;regularization_losses
knon_trainable_variables
llayer_metrics

mlayers
<trainable_variables
nmetrics
=	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
,
variable"
_generic_user_object
,
variable"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
olayer_regularization_losses
Fregularization_losses
pnon_trainable_variables
qlayer_metrics

rlayers
Gtrainable_variables
smetrics
H	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
,
variable"
_generic_user_object
,
variable"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
tlayer_regularization_losses
Qregularization_losses
unon_trainable_variables
vlayer_metrics

wlayers
Rtrainable_variables
xmetrics
S	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
,
variable"
_generic_user_object
,
variable"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
ylayer_regularization_losses
\regularization_losses
znon_trainable_variables
{layer_metrics

|layers
]trainable_variables
}metrics
^	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
?2?
!__inference__wrapped_model_200478?
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
&__inference_model_layer_call_fn_201819
&__inference_model_layer_call_fn_201973?
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
A__inference_model_layer_call_and_return_conditional_losses_201587
A__inference_model_layer_call_and_return_conditional_losses_201664
A__inference_model_layer_call_and_return_conditional_losses_202576
A__inference_model_layer_call_and_return_conditional_losses_203100?
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
*__inference_synthesis_layer_call_fn_201351
*__inference_synthesis_layer_call_fn_201509?
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
E__inference_synthesis_layer_call_and_return_conditional_losses_201105
E__inference_synthesis_layer_call_and_return_conditional_losses_201192
E__inference_synthesis_layer_call_and_return_conditional_losses_203624
E__inference_synthesis_layer_call_and_return_conditional_losses_204148?
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
$__inference_signature_wrapper_202052input_2"?
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
C__inference_layer_0_layer_call_and_return_conditional_losses_204317?
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
C__inference_layer_1_layer_call_and_return_conditional_losses_204486?
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
C__inference_layer_2_layer_call_and_return_conditional_losses_204655?
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
C__inference_layer_3_layer_call_and_return_conditional_losses_204698?
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
?2?
D__inference_lambda_1_layer_call_and_return_conditional_losses_204704
D__inference_lambda_1_layer_call_and_return_conditional_losses_204710?
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
!__inference__wrapped_model_200478?:??????????????????????K?H
A?>
<?9
input_2,????????????????????????????
? "O?L
J
	synthesis=?:
	synthesis+????????????????????????????
D__inference_lambda_1_layer_call_and_return_conditional_losses_204704?Q?N
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
D__inference_lambda_1_layer_call_and_return_conditional_losses_204710?Q?N
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
C__inference_layer_0_layer_call_and_return_conditional_losses_204317????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_1_layer_call_and_return_conditional_losses_204486????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_2_layer_call_and_return_conditional_losses_204655????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_3_layer_call_and_return_conditional_losses_204698??J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
A__inference_model_layer_call_and_return_conditional_losses_201587?:??????????????????????S?P
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
A__inference_model_layer_call_and_return_conditional_losses_201664?:??????????????????????S?P
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
A__inference_model_layer_call_and_return_conditional_losses_202576?:??????????????????????R?O
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
A__inference_model_layer_call_and_return_conditional_losses_203100?:??????????????????????R?O
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
&__inference_model_layer_call_fn_201819?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p

 
? "2?/+????????????????????????????
&__inference_model_layer_call_fn_201973?:??????????????????????S?P
I?F
<?9
input_2,????????????????????????????
p 

 
? "2?/+????????????????????????????
$__inference_signature_wrapper_202052?:??????????????????????V?S
? 
L?I
G
input_2<?9
input_2,????????????????????????????"O?L
J
	synthesis=?:
	synthesis+????????????????????????????
E__inference_synthesis_layer_call_and_return_conditional_losses_201105?:??????????????????????Y?V
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
E__inference_synthesis_layer_call_and_return_conditional_losses_201192?:??????????????????????Y?V
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
E__inference_synthesis_layer_call_and_return_conditional_losses_203624?:??????????????????????R?O
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
E__inference_synthesis_layer_call_and_return_conditional_losses_204148?:??????????????????????R?O
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
*__inference_synthesis_layer_call_fn_201351?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p

 
? "2?/+????????????????????????????
*__inference_synthesis_layer_call_fn_201509?:??????????????????????Y?V
O?L
B??
layer_0_input,????????????????????????????
p 

 
? "2?/+???????????????????????????