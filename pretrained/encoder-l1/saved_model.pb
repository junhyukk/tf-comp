??(
??
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.0-dev202103112v1.12.1-52612-g74d34665fc18??$
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
layer_0/gdn_0/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_0/gdn_0/reparam_beta
?
.layer_0/gdn_0/reparam_beta/Read/ReadVariableOpReadVariableOplayer_0/gdn_0/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_0/gdn_0/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelayer_0/gdn_0/reparam_gamma
?
/layer_0/gdn_0/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_0/gdn_0/reparam_gamma* 
_output_shapes
:
??*
dtype0
?
layer_0/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_namelayer_0/kernel_rdft
|
'layer_0/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_0/kernel_rdft*
_output_shapes
:	?*
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
layer_1/gdn_1/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_1/gdn_1/reparam_beta
?
.layer_1/gdn_1/reparam_beta/Read/ReadVariableOpReadVariableOplayer_1/gdn_1/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_1/gdn_1/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelayer_1/gdn_1/reparam_gamma
?
/layer_1/gdn_1/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_1/gdn_1/reparam_gamma* 
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
layer_2/gdn_2/reparam_betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*+
shared_namelayer_2/gdn_2/reparam_beta
?
.layer_2/gdn_2/reparam_beta/Read/ReadVariableOpReadVariableOplayer_2/gdn_2/reparam_beta*
_output_shapes	
:?*
dtype0
?
layer_2/gdn_2/reparam_gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*,
shared_namelayer_2/gdn_2/reparam_gamma
?
/layer_2/gdn_2/reparam_gamma/Read/ReadVariableOpReadVariableOplayer_2/gdn_2/reparam_gamma* 
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
q
layer_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namelayer_3/bias
j
 layer_3/bias/Read/ReadVariableOpReadVariableOplayer_3/bias*
_output_shapes	
:?*
dtype0
?
layer_3/kernel_rdftVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namelayer_3/kernel_rdft
}
'layer_3/kernel_rdft/Read/ReadVariableOpReadVariableOplayer_3/kernel_rdft* 
_output_shapes
:
??*
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
layer-0
	layer_with_weights-0
	layer-1

layer_with_weights-1

layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
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
R
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?
(_activation
)_kernel_parameter
_bias_parameter
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?
._activation
/_kernel_parameter
_bias_parameter
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?
4_activation
5_kernel_parameter
_bias_parameter
6regularization_losses
7trainable_variables
8	variables
9	keras_api
~
:_kernel_parameter
_bias_parameter
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
`^
VARIABLE_VALUElayer_0/gdn_0/reparam_beta0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElayer_0/gdn_0/reparam_gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer_0/kernel_rdft0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer_1/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElayer_1/gdn_1/reparam_beta0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUElayer_1/gdn_1/reparam_gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElayer_1/kernel_rdft0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUElayer_2/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUElayer_2/gdn_2/reparam_beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUElayer_2/gdn_2/reparam_gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
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
 
 
 
?
Dlayer_regularization_losses
$regularization_losses
Enon_trainable_variables
Flayer_metrics

Glayers
%trainable_variables
Hmetrics
&	variables
}
I_beta_parameter
J_gamma_parameter
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api


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
Olayer_regularization_losses
*regularization_losses
Pnon_trainable_variables
Qlayer_metrics

Rlayers
+trainable_variables
Smetrics
,	variables
}
T_beta_parameter
U_gamma_parameter
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api


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
Zlayer_regularization_losses
0regularization_losses
[non_trainable_variables
\layer_metrics

]layers
1trainable_variables
^metrics
2	variables
}
__beta_parameter
`_gamma_parameter
aregularization_losses
btrainable_variables
c	variables
d	keras_api


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
elayer_regularization_losses
6regularization_losses
fnon_trainable_variables
glayer_metrics

hlayers
7trainable_variables
imetrics
8	variables


rdft
 

0
1

0
1
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
 
 
 
 
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
Kregularization_losses
pnon_trainable_variables
qlayer_metrics

rlayers
Ltrainable_variables
smetrics
M	variables
 
 
 

(0
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
Vregularization_losses
unon_trainable_variables
vlayer_metrics

wlayers
Wtrainable_variables
xmetrics
X	variables
 
 
 

.0
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
aregularization_losses
znon_trainable_variables
{layer_metrics

|layers
btrainable_variables
}metrics
c	variables
 
 
 

40
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
serving_default_input_1Placeholder*A
_output_shapes/
-:+???????????????????????????*
dtype0*6
shape-:+???????????????????????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1Constlayer_0/kernel_rdftlayer_0/biasConst_1layer_0/gdn_0/reparam_gammaConst_2Const_3layer_0/gdn_0/reparam_betaConst_4Const_5Const_6Const_7layer_1/kernel_rdftlayer_1/biasConst_8layer_1/gdn_1/reparam_gammaConst_9Const_10layer_1/gdn_1/reparam_betaConst_11Const_12Const_13Const_14layer_2/kernel_rdftlayer_2/biasConst_15layer_2/gdn_2/reparam_gammaConst_16Const_17layer_2/gdn_2/reparam_betaConst_18Const_19Const_20Const_21layer_3/kernel_rdftlayer_3/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_197555
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename layer_0/bias/Read/ReadVariableOp.layer_0/gdn_0/reparam_beta/Read/ReadVariableOp/layer_0/gdn_0/reparam_gamma/Read/ReadVariableOp'layer_0/kernel_rdft/Read/ReadVariableOp layer_1/bias/Read/ReadVariableOp.layer_1/gdn_1/reparam_beta/Read/ReadVariableOp/layer_1/gdn_1/reparam_gamma/Read/ReadVariableOp'layer_1/kernel_rdft/Read/ReadVariableOp layer_2/bias/Read/ReadVariableOp.layer_2/gdn_2/reparam_beta/Read/ReadVariableOp/layer_2/gdn_2/reparam_gamma/Read/ReadVariableOp'layer_2/kernel_rdft/Read/ReadVariableOp layer_3/bias/Read/ReadVariableOp'layer_3/kernel_rdft/Read/ReadVariableOpConst_22*
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
__inference__traced_save_199830
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamelayer_0/biaslayer_0/gdn_0/reparam_betalayer_0/gdn_0/reparam_gammalayer_0/kernel_rdftlayer_1/biaslayer_1/gdn_1/reparam_betalayer_1/gdn_1/reparam_gammalayer_1/kernel_rdftlayer_2/biaslayer_2/gdn_2/reparam_betalayer_2/gdn_2/reparam_gammalayer_2/kernel_rdftlayer_3/biaslayer_3/kernel_rdft*
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
"__inference__traced_restore_199882??"
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_196091

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y?
truedivRealDivinputstruediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
truedivy
IdentityIdentitytruediv:z:0*
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
?
6encoder_analysis_layer_0_gdn_0_cond_1_cond_true_195695S
Oencoder_analysis_layer_0_gdn_0_cond_1_cond_abs_encoder_analysis_layer_0_biasadd:
6encoder_analysis_layer_0_gdn_0_cond_1_cond_placeholder7
3encoder_analysis_layer_0_gdn_0_cond_1_cond_identity?
.encoder/analysis/layer_0/gdn_0/cond_1/cond/AbsAbsOencoder_analysis_layer_0_gdn_0_cond_1_cond_abs_encoder_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_1/cond/Abs?
3encoder/analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity2encoder/analysis/layer_0/gdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity"s
3encoder_analysis_layer_0_gdn_0_cond_1_cond_identity<encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
*analysis_layer_0_gdn_0_cond_2_false_198092E
Aanalysis_layer_0_gdn_0_cond_2_cond_analysis_layer_0_gdn_0_biasadd)
%analysis_layer_0_gdn_0_cond_2_equal_x*
&analysis_layer_0_gdn_0_cond_2_identity?
analysis/layer_0/gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_0/gdn_0/cond_2/x?
#analysis/layer_0/gdn_0/cond_2/EqualEqual%analysis_layer_0_gdn_0_cond_2_equal_x(analysis/layer_0/gdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_0/gdn_0/cond_2/Equal?
"analysis/layer_0/gdn_0/cond_2/condStatelessIf'analysis/layer_0/gdn_0/cond_2/Equal:z:0Aanalysis_layer_0_gdn_0_cond_2_cond_analysis_layer_0_gdn_0_biasadd%analysis_layer_0_gdn_0_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_0_gdn_0_cond_2_cond_false_198101*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_2_cond_true_1981002$
"analysis/layer_0/gdn_0/cond_2/cond?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity"Y
&analysis_layer_0_gdn_0_cond_2_identity/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
1encoder_analysis_layer_2_gdn_2_cond_1_true_195958S
Oencoder_analysis_layer_2_gdn_2_cond_1_identity_encoder_analysis_layer_2_biasadd5
1encoder_analysis_layer_2_gdn_2_cond_1_placeholder2
.encoder_analysis_layer_2_gdn_2_cond_1_identity?
.encoder/analysis/layer_2/gdn_2/cond_1/IdentityIdentityOencoder_analysis_layer_2_gdn_2_cond_1_identity_encoder_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_1/Identity"i
.encoder_analysis_layer_2_gdn_2_cond_1_identity7encoder/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
gdn_2_cond_1_cond_true_196457!
gdn_2_cond_1_cond_abs_biasadd!
gdn_2_cond_1_cond_placeholder
gdn_2_cond_1_cond_identity?
gdn_2/cond_1/cond/AbsAbsgdn_2_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Abs?
gdn_2/cond_1/cond/IdentityIdentitygdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity"A
gdn_2_cond_1_cond_identity#gdn_2/cond_1/cond/Identity:output:0*(
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
.analysis_layer_0_gdn_0_cond_2_cond_true_197676J
Fanalysis_layer_0_gdn_0_cond_2_cond_sqrt_analysis_layer_0_gdn_0_biasadd2
.analysis_layer_0_gdn_0_cond_2_cond_placeholder/
+analysis_layer_0_gdn_0_cond_2_cond_identity?
'analysis/layer_0/gdn_0/cond_2/cond/SqrtSqrtFanalysis_layer_0_gdn_0_cond_2_cond_sqrt_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_0/gdn_0/cond_2/cond/Sqrt?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity"c
+analysis_layer_0_gdn_0_cond_2_cond_identity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
&layer_0_gdn_0_cond_2_cond_false_1985257
3layer_0_gdn_0_cond_2_cond_pow_layer_0_gdn_0_biasadd#
layer_0_gdn_0_cond_2_cond_pow_y&
"layer_0_gdn_0_cond_2_cond_identity?
layer_0/gdn_0/cond_2/cond/powPow3layer_0_gdn_0_cond_2_cond_pow_layer_0_gdn_0_biasaddlayer_0_gdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/cond/pow?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity!layer_0/gdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity"Q
"layer_0_gdn_0_cond_2_cond_identity+layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
(analysis_layer_0_gdn_0_cond_false_197998E
Aanalysis_layer_0_gdn_0_cond_identity_analysis_layer_0_gdn_0_equal
(
$analysis_layer_0_gdn_0_cond_identity
?
$analysis/layer_0/gdn_0/cond/IdentityIdentityAanalysis_layer_0_gdn_0_cond_identity_analysis_layer_0_gdn_0_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity"U
$analysis_layer_0_gdn_0_cond_identity-analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
M
gdn_2_cond_true_196437
gdn_2_cond_placeholder

gdn_2_cond_identity
f
gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_2/cond/Constr
gdn_2/cond/IdentityIdentitygdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity"3
gdn_2_cond_identitygdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
gdn_2_cond_2_cond_false_199675'
#gdn_2_cond_2_cond_pow_gdn_2_biasadd
gdn_2_cond_2_cond_pow_y
gdn_2_cond_2_cond_identity?
gdn_2/cond_2/cond/powPow#gdn_2_cond_2_cond_pow_gdn_2_biasaddgdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/pow?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity"A
gdn_2_cond_2_cond_identity#gdn_2/cond_2/cond/Identity:output:0*(
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
gdn_1_cond_1_cond_true_196293!
gdn_1_cond_1_cond_abs_biasadd!
gdn_1_cond_1_cond_placeholder
gdn_1_cond_1_cond_identity?
gdn_1/cond_1/cond/AbsAbsgdn_1_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Abs?
gdn_1/cond_1/cond/IdentityIdentitygdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity"A
gdn_1_cond_1_cond_identity#gdn_1/cond_1/cond/Identity:output:0*(
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
'analysis_layer_0_gdn_0_cond_true_197573+
'analysis_layer_0_gdn_0_cond_placeholder
(
$analysis_layer_0_gdn_0_cond_identity
?
!analysis/layer_0/gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_0/gdn_0/cond/Const?
$analysis/layer_0/gdn_0/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity"U
$analysis_layer_0_gdn_0_cond_identity-analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
*layer_0_gdn_0_cond_1_cond_cond_true_1988759
5layer_0_gdn_0_cond_1_cond_cond_square_layer_0_biasadd.
*layer_0_gdn_0_cond_1_cond_cond_placeholder+
'layer_0_gdn_0_cond_1_cond_cond_identity?
%layer_0/gdn_0/cond_1/cond/cond/SquareSquare5layer_0_gdn_0_cond_1_cond_cond_square_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_0/gdn_0/cond_1/cond/cond/Square?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity)layer_0/gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity"[
'layer_0_gdn_0_cond_1_cond_cond_identity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
|
gdn_2_cond_2_true_199665'
#gdn_2_cond_2_identity_gdn_2_biasadd
gdn_2_cond_2_placeholder
gdn_2_cond_2_identity?
gdn_2/cond_2/IdentityIdentity#gdn_2_cond_2_identity_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity"7
gdn_2_cond_2_identitygdn_2/cond_2/Identity:output:0*(
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
/encoder_analysis_layer_2_gdn_2_cond_true_1959473
/encoder_analysis_layer_2_gdn_2_cond_placeholder
0
,encoder_analysis_layer_2_gdn_2_cond_identity
?
)encoder/analysis/layer_2/gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)encoder/analysis/layer_2/gdn_2/cond/Const?
,encoder/analysis/layer_2/gdn_2/cond/IdentityIdentity2encoder/analysis/layer_2/gdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_2/gdn_2/cond/Identity"e
,encoder_analysis_layer_2_gdn_2_cond_identity5encoder/analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
.analysis_layer_2_gdn_2_cond_1_cond_true_198289C
?analysis_layer_2_gdn_2_cond_1_cond_abs_analysis_layer_2_biasadd2
.analysis_layer_2_gdn_2_cond_1_cond_placeholder/
+analysis_layer_2_gdn_2_cond_1_cond_identity?
&analysis/layer_2/gdn_2/cond_1/cond/AbsAbs?analysis_layer_2_gdn_2_cond_1_cond_abs_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/cond/Abs?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity"c
+analysis_layer_2_gdn_2_cond_1_cond_identity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
7encoder_analysis_layer_0_gdn_0_cond_2_cond_false_195779Y
Uencoder_analysis_layer_0_gdn_0_cond_2_cond_pow_encoder_analysis_layer_0_gdn_0_biasadd4
0encoder_analysis_layer_0_gdn_0_cond_2_cond_pow_y7
3encoder_analysis_layer_0_gdn_0_cond_2_cond_identity?
.encoder/analysis/layer_0/gdn_0/cond_2/cond/powPowUencoder_analysis_layer_0_gdn_0_cond_2_cond_pow_encoder_analysis_layer_0_gdn_0_biasadd0encoder_analysis_layer_0_gdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_2/cond/pow?
3encoder/analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity2encoder/analysis/layer_0/gdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity"s
3encoder_analysis_layer_0_gdn_0_cond_2_cond_identity<encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
.analysis_layer_1_gdn_1_cond_2_cond_true_198236J
Fanalysis_layer_1_gdn_1_cond_2_cond_sqrt_analysis_layer_1_gdn_1_biasadd2
.analysis_layer_1_gdn_1_cond_2_cond_placeholder/
+analysis_layer_1_gdn_1_cond_2_cond_identity?
'analysis/layer_1/gdn_1/cond_2/cond/SqrtSqrtFanalysis_layer_1_gdn_1_cond_2_cond_sqrt_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_1/gdn_1/cond_2/cond/Sqrt?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity"c
+analysis_layer_1_gdn_1_cond_2_cond_identity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
|
gdn_2_cond_2_true_196531'
#gdn_2_cond_2_identity_gdn_2_biasadd
gdn_2_cond_2_placeholder
gdn_2_cond_2_identity?
gdn_2/cond_2/IdentityIdentity#gdn_2_cond_2_identity_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity"7
gdn_2_cond_2_identitygdn_2/cond_2/Identity:output:0*(
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
&layer_1_gdn_1_cond_2_cond_false_1986617
3layer_1_gdn_1_cond_2_cond_pow_layer_1_gdn_1_biasadd#
layer_1_gdn_1_cond_2_cond_pow_y&
"layer_1_gdn_1_cond_2_cond_identity?
layer_1/gdn_1/cond_2/cond/powPow3layer_1_gdn_1_cond_2_cond_pow_layer_1_gdn_1_biasaddlayer_1_gdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/cond/pow?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity!layer_1/gdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity"Q
"layer_1_gdn_1_cond_2_cond_identity+layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
D__inference_analysis_layer_call_and_return_conditional_losses_196608
lambda_input
layer_0_196234!
layer_0_196236:	?
layer_0_196238:	?
layer_0_196240"
layer_0_196242:
??
layer_0_196244
layer_0_196246
layer_0_196248:	?
layer_0_196250
layer_0_196252
layer_0_196254
layer_1_196398"
layer_1_196400:
??
layer_1_196402:	?
layer_1_196404"
layer_1_196406:
??
layer_1_196408
layer_1_196410
layer_1_196412:	?
layer_1_196414
layer_1_196416
layer_1_196418
layer_2_196562"
layer_2_196564:
??
layer_2_196566:	?
layer_2_196568"
layer_2_196570:
??
layer_2_196572
layer_2_196574
layer_2_196576:	?
layer_2_196578
layer_2_196580
layer_2_196582
layer_3_196600"
layer_3_196602:
??
layer_3_196604:	?
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
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
GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_1960912
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196234layer_0_196236layer_0_196238layer_0_196240layer_0_196242layer_0_196244layer_0_196246layer_0_196248layer_0_196250layer_0_196252layer_0_196254*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962332!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196398layer_1_196400layer_1_196402layer_1_196404layer_1_196406layer_1_196408layer_1_196410layer_1_196412layer_1_196414layer_1_196416layer_1_196418*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1963972!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196562layer_2_196564layer_2_196566layer_2_196568layer_2_196570layer_2_196572layer_2_196574layer_2_196576layer_2_196578layer_2_196580layer_2_196582*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1965612!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196600layer_3_196602layer_3_196604*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_1965992!
layer_3/StatefulPartitionedCall?
IdentityIdentity(layer_3/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:o k
A
_output_shapes/
-:+???????????????????????????
&
_user_specified_namelambda_input:$ 

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
gdn_0_cond_1_cond_true_199303!
gdn_0_cond_1_cond_abs_biasadd!
gdn_0_cond_1_cond_placeholder
gdn_0_cond_1_cond_identity?
gdn_0/cond_1/cond/AbsAbsgdn_0_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Abs?
gdn_0/cond_1/cond/IdentityIdentitygdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity"A
gdn_0_cond_1_cond_identity#gdn_0/cond_1/cond/Identity:output:0*(
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
D__inference_analysis_layer_call_and_return_conditional_losses_196779

inputs
layer_0_196702!
layer_0_196704:	?
layer_0_196706:	?
layer_0_196708"
layer_0_196710:
??
layer_0_196712
layer_0_196714
layer_0_196716:	?
layer_0_196718
layer_0_196720
layer_0_196722
layer_1_196725"
layer_1_196727:
??
layer_1_196729:	?
layer_1_196731"
layer_1_196733:
??
layer_1_196735
layer_1_196737
layer_1_196739:	?
layer_1_196741
layer_1_196743
layer_1_196745
layer_2_196748"
layer_2_196750:
??
layer_2_196752:	?
layer_2_196754"
layer_2_196756:
??
layer_2_196758
layer_2_196760
layer_2_196762:	?
layer_2_196764
layer_2_196766
layer_2_196768
layer_3_196771"
layer_3_196773:
??
layer_3_196775:	?
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_1960912
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196702layer_0_196704layer_0_196706layer_0_196708layer_0_196710layer_0_196712layer_0_196714layer_0_196716layer_0_196718layer_0_196720layer_0_196722*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962332!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196725layer_1_196727layer_1_196729layer_1_196731layer_1_196733layer_1_196735layer_1_196737layer_1_196739layer_1_196741layer_1_196743layer_1_196745*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1963972!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196748layer_2_196750layer_2_196752layer_2_196754layer_2_196756layer_2_196758layer_2_196760layer_2_196762layer_2_196764layer_2_196766layer_2_196768*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1965612!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196771layer_3_196773layer_3_196775*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_1965992!
layer_3/StatefulPartitionedCall?
IdentityIdentity(layer_3/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
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
#gdn_2_cond_1_cond_cond_false_196468&
"gdn_2_cond_1_cond_cond_pow_biasadd 
gdn_2_cond_1_cond_cond_pow_y#
gdn_2_cond_1_cond_cond_identity?
gdn_2/cond_1/cond/cond/powPow"gdn_2_cond_1_cond_cond_pow_biasaddgdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/cond/pow?
gdn_2/cond_1/cond/cond/IdentityIdentitygdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity"K
gdn_2_cond_1_cond_cond_identity(gdn_2/cond_1/cond/cond/Identity:output:0*(
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
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_197604H
Danalysis_layer_0_gdn_0_cond_1_cond_cond_pow_analysis_layer_0_biasadd1
-analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y4
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
+analysis/layer_0/gdn_0/cond_1/cond/cond/powPowDanalysis_layer_0_gdn_0_cond_1_cond_cond_pow_analysis_layer_0_biasadd-analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/cond/pow?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity/analysis/layer_0/gdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"m
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
/analysis_layer_0_gdn_0_cond_2_cond_false_198101I
Eanalysis_layer_0_gdn_0_cond_2_cond_pow_analysis_layer_0_gdn_0_biasadd,
(analysis_layer_0_gdn_0_cond_2_cond_pow_y/
+analysis_layer_0_gdn_0_cond_2_cond_identity?
&analysis/layer_0/gdn_0/cond_2/cond/powPowEanalysis_layer_0_gdn_0_cond_2_cond_pow_analysis_layer_0_gdn_0_biasadd(analysis_layer_0_gdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/cond/pow?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity"c
+analysis_layer_0_gdn_0_cond_2_cond_identity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
e
layer_0_gdn_0_cond_true_198421"
layer_0_gdn_0_cond_placeholder

layer_0_gdn_0_cond_identity
v
layer_0/gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_0/gdn_0/cond/Const?
layer_0/gdn_0/cond/IdentityIdentity!layer_0/gdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity"C
layer_0_gdn_0_cond_identity$layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
)__inference_analysis_layer_call_fn_197012
lambda_input
unknown
	unknown_0:	?
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

unknown_33:
??

unknown_34:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1969372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:o k
A
_output_shapes/
-:+???????????????????????????
&
_user_specified_namelambda_input:$ 

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
%layer_1_gdn_1_cond_2_cond_true_1986608
4layer_1_gdn_1_cond_2_cond_sqrt_layer_1_gdn_1_biasadd)
%layer_1_gdn_1_cond_2_cond_placeholder&
"layer_1_gdn_1_cond_2_cond_identity?
layer_1/gdn_1/cond_2/cond/SqrtSqrt4layer_1_gdn_1_cond_2_cond_sqrt_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/gdn_1/cond_2/cond/Sqrt?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity"layer_1/gdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity"Q
"layer_1_gdn_1_cond_2_cond_identity+layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_198027K
Ganalysis_layer_0_gdn_0_cond_1_cond_cond_square_analysis_layer_0_biasadd7
3analysis_layer_0_gdn_0_cond_1_cond_cond_placeholder4
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
.analysis/layer_0/gdn_0/cond_1/cond/cond/SquareSquareGanalysis_layer_0_gdn_0_cond_1_cond_cond_square_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_0/gdn_0/cond_1/cond/cond/Square?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity2analysis/layer_0/gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"m
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
w
layer_0_gdn_0_cond_false_1984223
/layer_0_gdn_0_cond_identity_layer_0_gdn_0_equal

layer_0_gdn_0_cond_identity
?
layer_0/gdn_0/cond/IdentityIdentity/layer_0_gdn_0_cond_identity_layer_0_gdn_0_equal*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity"C
layer_0_gdn_0_cond_identity$layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
|
gdn_1_cond_2_true_196367'
#gdn_1_cond_2_identity_gdn_1_biasadd
gdn_1_cond_2_placeholder
gdn_1_cond_2_identity?
gdn_1/cond_2/IdentityIdentity#gdn_1_cond_2_identity_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity"7
gdn_1_cond_2_identitygdn_1/cond_2/Identity:output:0*(
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
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_198299K
Ganalysis_layer_2_gdn_2_cond_1_cond_cond_square_analysis_layer_2_biasadd7
3analysis_layer_2_gdn_2_cond_1_cond_cond_placeholder4
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
.analysis/layer_2/gdn_2/cond_1/cond/cond/SquareSquareGanalysis_layer_2_gdn_2_cond_1_cond_cond_square_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_2/gdn_2/cond_1/cond/cond/Square?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity2analysis/layer_2/gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"m
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
)analysis_layer_1_gdn_1_cond_2_true_197803I
Eanalysis_layer_1_gdn_1_cond_2_identity_analysis_layer_1_gdn_1_biasadd-
)analysis_layer_1_gdn_1_cond_2_placeholder*
&analysis_layer_1_gdn_1_cond_2_identity?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentityEanalysis_layer_1_gdn_1_cond_2_identity_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity"Y
&analysis_layer_1_gdn_1_cond_2_identity/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
!layer_2_gdn_2_cond_2_false_1987883
/layer_2_gdn_2_cond_2_cond_layer_2_gdn_2_biasadd 
layer_2_gdn_2_cond_2_equal_x!
layer_2_gdn_2_cond_2_identityu
layer_2/gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_2/gdn_2/cond_2/x?
layer_2/gdn_2/cond_2/EqualEquallayer_2_gdn_2_cond_2_equal_xlayer_2/gdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/cond_2/Equal?
layer_2/gdn_2/cond_2/condStatelessIflayer_2/gdn_2/cond_2/Equal:z:0/layer_2_gdn_2_cond_2_cond_layer_2_gdn_2_biasaddlayer_2_gdn_2_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_2_gdn_2_cond_2_cond_false_198797*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_2_cond_true_1987962
layer_2/gdn_2/cond_2/cond?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity"layer_2/gdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity?
layer_2/gdn_2/cond_2/IdentityIdentity+layer_2/gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity"G
layer_2_gdn_2_cond_2_identity&layer_2/gdn_2/cond_2/Identity:output:0*(
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
!layer_1_gdn_1_cond_1_false_198993-
)layer_1_gdn_1_cond_1_cond_layer_1_biasadd 
layer_1_gdn_1_cond_1_equal_x!
layer_1_gdn_1_cond_1_identityu
layer_1/gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/cond_1/x?
layer_1/gdn_1/cond_1/EqualEquallayer_1_gdn_1_cond_1_equal_xlayer_1/gdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/cond_1/Equal?
layer_1/gdn_1/cond_1/condStatelessIflayer_1/gdn_1/cond_1/Equal:z:0)layer_1_gdn_1_cond_1_cond_layer_1_biasaddlayer_1_gdn_1_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_1_gdn_1_cond_1_cond_false_199002*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_1_cond_true_1990012
layer_1/gdn_1/cond_1/cond?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity"layer_1/gdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity?
layer_1/gdn_1/cond_1/IdentityIdentity+layer_1/gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity"G
layer_1_gdn_1_cond_1_identity&layer_1/gdn_1/cond_1/Identity:output:0*(
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
&layer_0_gdn_0_cond_2_cond_false_1989497
3layer_0_gdn_0_cond_2_cond_pow_layer_0_gdn_0_biasadd#
layer_0_gdn_0_cond_2_cond_pow_y&
"layer_0_gdn_0_cond_2_cond_identity?
layer_0/gdn_0/cond_2/cond/powPow3layer_0_gdn_0_cond_2_cond_pow_layer_0_gdn_0_biasaddlayer_0_gdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/cond/pow?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity!layer_0/gdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity"Q
"layer_0_gdn_0_cond_2_cond_identity+layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
?
7encoder_analysis_layer_2_gdn_2_cond_1_cond_false_195968T
Pencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_encoder_analysis_layer_2_biasadd6
2encoder_analysis_layer_2_gdn_2_cond_1_cond_equal_x7
3encoder_analysis_layer_2_gdn_2_cond_1_cond_identity?
,encoder/analysis/layer_2/gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,encoder/analysis/layer_2/gdn_2/cond_1/cond/x?
0encoder/analysis/layer_2/gdn_2/cond_1/cond/EqualEqual2encoder_analysis_layer_2_gdn_2_cond_1_cond_equal_x5encoder/analysis/layer_2/gdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0encoder/analysis/layer_2/gdn_2/cond_1/cond/Equal?
/encoder/analysis/layer_2/gdn_2/cond_1/cond/condStatelessIf4encoder/analysis/layer_2/gdn_2/cond_1/cond/Equal:z:0Pencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_encoder_analysis_layer_2_biasadd2encoder_analysis_layer_2_gdn_2_cond_1_cond_equal_x*
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
<encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_false_195978*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_true_19597721
/encoder/analysis/layer_2/gdn_2/cond_1/cond/cond?
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity?
3encoder/analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentityAencoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity"s
3encoder_analysis_layer_2_gdn_2_cond_1_cond_identity<encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
*analysis_layer_1_gdn_1_cond_2_false_197804E
Aanalysis_layer_1_gdn_1_cond_2_cond_analysis_layer_1_gdn_1_biasadd)
%analysis_layer_1_gdn_1_cond_2_equal_x*
&analysis_layer_1_gdn_1_cond_2_identity?
analysis/layer_1/gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_1/gdn_1/cond_2/x?
#analysis/layer_1/gdn_1/cond_2/EqualEqual%analysis_layer_1_gdn_1_cond_2_equal_x(analysis/layer_1/gdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_1/gdn_1/cond_2/Equal?
"analysis/layer_1/gdn_1/cond_2/condStatelessIf'analysis/layer_1/gdn_1/cond_2/Equal:z:0Aanalysis_layer_1_gdn_1_cond_2_cond_analysis_layer_1_gdn_1_biasadd%analysis_layer_1_gdn_1_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_1_gdn_1_cond_2_cond_false_197813*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_2_cond_true_1978122$
"analysis/layer_1/gdn_1/cond_2/cond?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity"Y
&analysis_layer_1_gdn_1_cond_2_identity/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
6encoder_analysis_layer_1_gdn_1_cond_2_cond_true_195914Z
Vencoder_analysis_layer_1_gdn_1_cond_2_cond_sqrt_encoder_analysis_layer_1_gdn_1_biasadd:
6encoder_analysis_layer_1_gdn_1_cond_2_cond_placeholder7
3encoder_analysis_layer_1_gdn_1_cond_2_cond_identity?
/encoder/analysis/layer_1/gdn_1/cond_2/cond/SqrtSqrtVencoder_analysis_layer_1_gdn_1_cond_2_cond_sqrt_encoder_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/encoder/analysis/layer_1/gdn_1/cond_2/cond/Sqrt?
3encoder/analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity3encoder/analysis/layer_1/gdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity"s
3encoder_analysis_layer_1_gdn_1_cond_2_cond_identity<encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
2encoder_analysis_layer_2_gdn_2_cond_2_false_196042U
Qencoder_analysis_layer_2_gdn_2_cond_2_cond_encoder_analysis_layer_2_gdn_2_biasadd1
-encoder_analysis_layer_2_gdn_2_cond_2_equal_x2
.encoder_analysis_layer_2_gdn_2_cond_2_identity?
'encoder/analysis/layer_2/gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder/analysis/layer_2/gdn_2/cond_2/x?
+encoder/analysis/layer_2/gdn_2/cond_2/EqualEqual-encoder_analysis_layer_2_gdn_2_cond_2_equal_x0encoder/analysis/layer_2/gdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_2/gdn_2/cond_2/Equal?
*encoder/analysis/layer_2/gdn_2/cond_2/condStatelessIf/encoder/analysis/layer_2/gdn_2/cond_2/Equal:z:0Qencoder_analysis_layer_2_gdn_2_cond_2_cond_encoder_analysis_layer_2_gdn_2_biasadd-encoder_analysis_layer_2_gdn_2_cond_2_equal_x*
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
7encoder_analysis_layer_2_gdn_2_cond_2_cond_false_196051*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_2_gdn_2_cond_2_cond_true_1960502,
*encoder/analysis/layer_2/gdn_2/cond_2/cond?
3encoder/analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity3encoder/analysis/layer_2/gdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity?
.encoder/analysis/layer_2/gdn_2/cond_2/IdentityIdentity<encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_2/Identity"i
.encoder_analysis_layer_2_gdn_2_cond_2_identity7encoder/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
 layer_0_gdn_0_cond_2_true_1985157
3layer_0_gdn_0_cond_2_identity_layer_0_gdn_0_biasadd$
 layer_0_gdn_0_cond_2_placeholder!
layer_0_gdn_0_cond_2_identity?
layer_0/gdn_0/cond_2/IdentityIdentity3layer_0_gdn_0_cond_2_identity_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity"G
layer_0_gdn_0_cond_2_identity&layer_0/gdn_0/cond_2/Identity:output:0*(
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
!layer_1_gdn_1_cond_1_false_198569-
)layer_1_gdn_1_cond_1_cond_layer_1_biasadd 
layer_1_gdn_1_cond_1_equal_x!
layer_1_gdn_1_cond_1_identityu
layer_1/gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/cond_1/x?
layer_1/gdn_1/cond_1/EqualEquallayer_1_gdn_1_cond_1_equal_xlayer_1/gdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/cond_1/Equal?
layer_1/gdn_1/cond_1/condStatelessIflayer_1/gdn_1/cond_1/Equal:z:0)layer_1_gdn_1_cond_1_cond_layer_1_biasaddlayer_1_gdn_1_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_1_gdn_1_cond_1_cond_false_198578*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_1_cond_true_1985772
layer_1/gdn_1/cond_1/cond?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity"layer_1/gdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity?
layer_1/gdn_1/cond_1/IdentityIdentity+layer_1/gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity"G
layer_1_gdn_1_cond_1_identity&layer_1/gdn_1/cond_1/Identity:output:0*(
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
.analysis_layer_0_gdn_0_cond_2_cond_true_198100J
Fanalysis_layer_0_gdn_0_cond_2_cond_sqrt_analysis_layer_0_gdn_0_biasadd2
.analysis_layer_0_gdn_0_cond_2_cond_placeholder/
+analysis_layer_0_gdn_0_cond_2_cond_identity?
'analysis/layer_0/gdn_0/cond_2/cond/SqrtSqrtFanalysis_layer_0_gdn_0_cond_2_cond_sqrt_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_0/gdn_0/cond_2/cond/Sqrt?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity"c
+analysis_layer_0_gdn_0_cond_2_cond_identity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
.analysis_layer_1_gdn_1_cond_1_cond_true_197729C
?analysis_layer_1_gdn_1_cond_1_cond_abs_analysis_layer_1_biasadd2
.analysis_layer_1_gdn_1_cond_1_cond_placeholder/
+analysis_layer_1_gdn_1_cond_1_cond_identity?
&analysis/layer_1/gdn_1/cond_1/cond/AbsAbs?analysis_layer_1_gdn_1_cond_1_cond_abs_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/cond/Abs?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity"c
+analysis_layer_1_gdn_1_cond_1_cond_identity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
"gdn_2_cond_1_cond_cond_true_199601)
%gdn_2_cond_1_cond_cond_square_biasadd&
"gdn_2_cond_1_cond_cond_placeholder#
gdn_2_cond_1_cond_cond_identity?
gdn_2/cond_1/cond/cond/SquareSquare%gdn_2_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/cond/Square?
gdn_2/cond_1/cond/cond/IdentityIdentity!gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity"K
gdn_2_cond_1_cond_cond_identity(gdn_2/cond_1/cond/cond/Identity:output:0*(
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
?
C__inference_layer_3_layer_call_and_return_conditional_losses_196599

inputs
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
Conv2DConv2Dinputslayer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

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
?
?
)__inference_analysis_layer_call_fn_196854
lambda_input
unknown
	unknown_0:	?
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

unknown_33:
??

unknown_34:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalllambda_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1967792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:o k
A
_output_shapes/
-:+???????????????????????????
&
_user_specified_namelambda_input:$ 

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
%layer_2_gdn_2_cond_2_cond_true_1992208
4layer_2_gdn_2_cond_2_cond_sqrt_layer_2_gdn_2_biasadd)
%layer_2_gdn_2_cond_2_cond_placeholder&
"layer_2_gdn_2_cond_2_cond_identity?
layer_2/gdn_2/cond_2/cond/SqrtSqrt4layer_2_gdn_2_cond_2_cond_sqrt_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/gdn_2/cond_2/cond/Sqrt?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity"layer_2/gdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity"Q
"layer_2_gdn_2_cond_2_cond_identity+layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
!layer_0_gdn_0_cond_2_false_1985163
/layer_0_gdn_0_cond_2_cond_layer_0_gdn_0_biasadd 
layer_0_gdn_0_cond_2_equal_x!
layer_0_gdn_0_cond_2_identityu
layer_0/gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_0/gdn_0/cond_2/x?
layer_0/gdn_0/cond_2/EqualEquallayer_0_gdn_0_cond_2_equal_xlayer_0/gdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/cond_2/Equal?
layer_0/gdn_0/cond_2/condStatelessIflayer_0/gdn_0/cond_2/Equal:z:0/layer_0_gdn_0_cond_2_cond_layer_0_gdn_0_biasaddlayer_0_gdn_0_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_0_gdn_0_cond_2_cond_false_198525*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_2_cond_true_1985242
layer_0/gdn_0/cond_2/cond?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity"layer_0/gdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity?
layer_0/gdn_0/cond_2/IdentityIdentity+layer_0/gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity"G
layer_0_gdn_0_cond_2_identity&layer_0/gdn_0/cond_2/Identity:output:0*(
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
W
gdn_0_cond_false_196110#
gdn_0_cond_identity_gdn_0_equal

gdn_0_cond_identity
x
gdn_0/cond/IdentityIdentitygdn_0_cond_identity_gdn_0_equal*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity"3
gdn_0_cond_identitygdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
"gdn_2_cond_1_cond_cond_true_196467)
%gdn_2_cond_1_cond_cond_square_biasadd&
"gdn_2_cond_1_cond_cond_placeholder#
gdn_2_cond_1_cond_cond_identity?
gdn_2/cond_1/cond/cond/SquareSquare%gdn_2_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/cond/Square?
gdn_2/cond_1/cond/cond/IdentityIdentity!gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity"K
gdn_2_cond_1_cond_cond_identity(gdn_2/cond_1/cond/cond/Identity:output:0*(
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
.analysis_layer_0_gdn_0_cond_1_cond_true_198017C
?analysis_layer_0_gdn_0_cond_1_cond_abs_analysis_layer_0_biasadd2
.analysis_layer_0_gdn_0_cond_1_cond_placeholder/
+analysis_layer_0_gdn_0_cond_1_cond_identity?
&analysis/layer_0/gdn_0/cond_1/cond/AbsAbs?analysis_layer_0_gdn_0_cond_1_cond_abs_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/cond/Abs?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity"c
+analysis_layer_0_gdn_0_cond_1_cond_identity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
W
gdn_0_cond_false_199284#
gdn_0_cond_identity_gdn_0_equal

gdn_0_cond_identity
x
gdn_0/cond/IdentityIdentitygdn_0_cond_identity_gdn_0_equal*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity"3
gdn_0_cond_identitygdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
v
gdn_1_cond_1_true_196284!
gdn_1_cond_1_identity_biasadd
gdn_1_cond_1_placeholder
gdn_1_cond_1_identity?
gdn_1/cond_1/IdentityIdentitygdn_1_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity"7
gdn_1_cond_1_identitygdn_1/cond_1/Identity:output:0*(
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
/analysis_layer_0_gdn_0_cond_2_cond_false_197677I
Eanalysis_layer_0_gdn_0_cond_2_cond_pow_analysis_layer_0_gdn_0_biasadd,
(analysis_layer_0_gdn_0_cond_2_cond_pow_y/
+analysis_layer_0_gdn_0_cond_2_cond_identity?
&analysis/layer_0/gdn_0/cond_2/cond/powPowEanalysis_layer_0_gdn_0_cond_2_cond_pow_analysis_layer_0_gdn_0_biasadd(analysis_layer_0_gdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/cond/pow?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity"c
+analysis_layer_0_gdn_0_cond_2_cond_identity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
6encoder_analysis_layer_2_gdn_2_cond_2_cond_true_196050Z
Vencoder_analysis_layer_2_gdn_2_cond_2_cond_sqrt_encoder_analysis_layer_2_gdn_2_biasadd:
6encoder_analysis_layer_2_gdn_2_cond_2_cond_placeholder7
3encoder_analysis_layer_2_gdn_2_cond_2_cond_identity?
/encoder/analysis/layer_2/gdn_2/cond_2/cond/SqrtSqrtVencoder_analysis_layer_2_gdn_2_cond_2_cond_sqrt_encoder_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/encoder/analysis/layer_2/gdn_2/cond_2/cond/Sqrt?
3encoder/analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity3encoder/analysis/layer_2/gdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity"s
3encoder_analysis_layer_2_gdn_2_cond_2_cond_identity<encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
)analysis_layer_2_gdn_2_cond_2_true_197939I
Eanalysis_layer_2_gdn_2_cond_2_identity_analysis_layer_2_gdn_2_biasadd-
)analysis_layer_2_gdn_2_cond_2_placeholder*
&analysis_layer_2_gdn_2_cond_2_identity?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentityEanalysis_layer_2_gdn_2_cond_2_identity_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity"Y
&analysis_layer_2_gdn_2_cond_2_identity/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
/analysis_layer_2_gdn_2_cond_2_cond_false_197949I
Eanalysis_layer_2_gdn_2_cond_2_cond_pow_analysis_layer_2_gdn_2_biasadd,
(analysis_layer_2_gdn_2_cond_2_cond_pow_y/
+analysis_layer_2_gdn_2_cond_2_cond_identity?
&analysis/layer_2/gdn_2/cond_2/cond/powPowEanalysis_layer_2_gdn_2_cond_2_cond_pow_analysis_layer_2_gdn_2_biasadd(analysis_layer_2_gdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/cond/pow?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity"c
+analysis_layer_2_gdn_2_cond_2_cond_identity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
)analysis_layer_1_gdn_1_cond_1_true_198144C
?analysis_layer_1_gdn_1_cond_1_identity_analysis_layer_1_biasadd-
)analysis_layer_1_gdn_1_cond_1_placeholder*
&analysis_layer_1_gdn_1_cond_1_identity?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity?analysis_layer_1_gdn_1_cond_1_identity_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity"Y
&analysis_layer_1_gdn_1_cond_1_identity/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
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
gdn_0_cond_1_cond_false_196130"
gdn_0_cond_1_cond_cond_biasadd
gdn_0_cond_1_cond_equal_x
gdn_0_cond_1_cond_identityo
gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_0/cond_1/cond/x?
gdn_0/cond_1/cond/EqualEqualgdn_0_cond_1_cond_equal_xgdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_1/cond/Equal?
gdn_0/cond_1/cond/condStatelessIfgdn_0/cond_1/cond/Equal:z:0gdn_0_cond_1_cond_cond_biasaddgdn_0_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_0_cond_1_cond_cond_false_196140*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_0_cond_1_cond_cond_true_1961392
gdn_0/cond_1/cond/cond?
gdn_0/cond_1/cond/cond/IdentityIdentitygdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity?
gdn_0/cond_1/cond/IdentityIdentity(gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity"A
gdn_0_cond_1_cond_identity#gdn_0/cond_1/cond/Identity:output:0*(
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
 layer_0_gdn_0_cond_2_true_1989397
3layer_0_gdn_0_cond_2_identity_layer_0_gdn_0_biasadd$
 layer_0_gdn_0_cond_2_placeholder!
layer_0_gdn_0_cond_2_identity?
layer_0/gdn_0/cond_2/IdentityIdentity3layer_0_gdn_0_cond_2_identity_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity"G
layer_0_gdn_0_cond_2_identity&layer_0/gdn_0/cond_2/Identity:output:0*(
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
&layer_2_gdn_2_cond_1_cond_false_1991382
.layer_2_gdn_2_cond_1_cond_cond_layer_2_biasadd%
!layer_2_gdn_2_cond_1_cond_equal_x&
"layer_2_gdn_2_cond_1_cond_identity
layer_2/gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_2/gdn_2/cond_1/cond/x?
layer_2/gdn_2/cond_1/cond/EqualEqual!layer_2_gdn_2_cond_1_cond_equal_x$layer_2/gdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_2/gdn_2/cond_1/cond/Equal?
layer_2/gdn_2/cond_1/cond/condStatelessIf#layer_2/gdn_2/cond_1/cond/Equal:z:0.layer_2_gdn_2_cond_1_cond_cond_layer_2_biasadd!layer_2_gdn_2_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_2_gdn_2_cond_1_cond_cond_false_199148*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_2_gdn_2_cond_1_cond_cond_true_1991472 
layer_2/gdn_2/cond_1/cond/cond?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity'layer_2/gdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity"Q
"layer_2_gdn_2_cond_1_cond_identity+layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
D__inference_analysis_layer_call_and_return_conditional_losses_196937

inputs
layer_0_196860!
layer_0_196862:	?
layer_0_196864:	?
layer_0_196866"
layer_0_196868:
??
layer_0_196870
layer_0_196872
layer_0_196874:	?
layer_0_196876
layer_0_196878
layer_0_196880
layer_1_196883"
layer_1_196885:
??
layer_1_196887:	?
layer_1_196889"
layer_1_196891:
??
layer_1_196893
layer_1_196895
layer_1_196897:	?
layer_1_196899
layer_1_196901
layer_1_196903
layer_2_196906"
layer_2_196908:
??
layer_2_196910:	?
layer_2_196912"
layer_2_196914:
??
layer_2_196916
layer_2_196918
layer_2_196920:	?
layer_2_196922
layer_2_196924
layer_2_196926
layer_3_196929"
layer_3_196931:
??
layer_3_196933:	?
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCallinputs*
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
GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_1966162
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196860layer_0_196862layer_0_196864layer_0_196866layer_0_196868layer_0_196870layer_0_196872layer_0_196874layer_0_196876layer_0_196878layer_0_196880*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962332!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196883layer_1_196885layer_1_196887layer_1_196889layer_1_196891layer_1_196893layer_1_196895layer_1_196897layer_1_196899layer_1_196901layer_1_196903*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1963972!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196906layer_2_196908layer_2_196910layer_2_196912layer_2_196914layer_2_196916layer_2_196918layer_2_196920layer_2_196922layer_2_196924layer_2_196926*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1965612!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196929layer_3_196931layer_3_196933*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_1965992!
layer_3/StatefulPartitionedCall?
IdentityIdentity(layer_3/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
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
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_198300H
Danalysis_layer_2_gdn_2_cond_1_cond_cond_pow_analysis_layer_2_biasadd1
-analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y4
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
+analysis/layer_2/gdn_2/cond_1/cond/cond/powPowDanalysis_layer_2_gdn_2_cond_1_cond_cond_pow_analysis_layer_2_biasadd-analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/cond/pow?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity/analysis/layer_2/gdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"m
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
gdn_1_cond_2_cond_true_199530(
$gdn_1_cond_2_cond_sqrt_gdn_1_biasadd!
gdn_1_cond_2_cond_placeholder
gdn_1_cond_2_cond_identity?
gdn_1/cond_2/cond/SqrtSqrt$gdn_1_cond_2_cond_sqrt_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Sqrt?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity"A
gdn_1_cond_2_cond_identity#gdn_1/cond_2/cond/Identity:output:0*(
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
v
gdn_1_cond_1_true_199438!
gdn_1_cond_1_identity_biasadd
gdn_1_cond_1_placeholder
gdn_1_cond_1_identity?
gdn_1/cond_1/IdentityIdentitygdn_1_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity"7
gdn_1_cond_1_identitygdn_1/cond_1/Identity:output:0*(
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
*analysis_layer_1_gdn_1_cond_1_false_197721?
;analysis_layer_1_gdn_1_cond_1_cond_analysis_layer_1_biasadd)
%analysis_layer_1_gdn_1_cond_1_equal_x*
&analysis_layer_1_gdn_1_cond_1_identity?
analysis/layer_1/gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_1/gdn_1/cond_1/x?
#analysis/layer_1/gdn_1/cond_1/EqualEqual%analysis_layer_1_gdn_1_cond_1_equal_x(analysis/layer_1/gdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_1/gdn_1/cond_1/Equal?
"analysis/layer_1/gdn_1/cond_1/condStatelessIf'analysis/layer_1/gdn_1/cond_1/Equal:z:0;analysis_layer_1_gdn_1_cond_1_cond_analysis_layer_1_biasadd%analysis_layer_1_gdn_1_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_1_gdn_1_cond_1_cond_false_197730*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_1_cond_true_1977292$
"analysis/layer_1/gdn_1/cond_1/cond?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity"Y
&analysis_layer_1_gdn_1_cond_1_identity/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
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
o
gdn_1_cond_1_false_199439
gdn_1_cond_1_cond_biasadd
gdn_1_cond_1_equal_x
gdn_1_cond_1_identitye
gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_1/cond_1/x?
gdn_1/cond_1/EqualEqualgdn_1_cond_1_equal_xgdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_1/Equal?
gdn_1/cond_1/condStatelessIfgdn_1/cond_1/Equal:z:0gdn_1_cond_1_cond_biasaddgdn_1_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_1_cond_1_cond_false_199448*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_1_cond_true_1994472
gdn_1/cond_1/cond?
gdn_1/cond_1/cond/IdentityIdentitygdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity?
gdn_1/cond_1/IdentityIdentity#gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity"7
gdn_1_cond_1_identitygdn_1/cond_1/Identity:output:0*(
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
D__inference_analysis_layer_call_and_return_conditional_losses_196695
lambda_input
layer_0_196618!
layer_0_196620:	?
layer_0_196622:	?
layer_0_196624"
layer_0_196626:
??
layer_0_196628
layer_0_196630
layer_0_196632:	?
layer_0_196634
layer_0_196636
layer_0_196638
layer_1_196641"
layer_1_196643:
??
layer_1_196645:	?
layer_1_196647"
layer_1_196649:
??
layer_1_196651
layer_1_196653
layer_1_196655:	?
layer_1_196657
layer_1_196659
layer_1_196661
layer_2_196664"
layer_2_196666:
??
layer_2_196668:	?
layer_2_196670"
layer_2_196672:
??
layer_2_196674
layer_2_196676
layer_2_196678:	?
layer_2_196680
layer_2_196682
layer_2_196684
layer_3_196687"
layer_3_196689:
??
layer_3_196691:	?
identity??layer_0/StatefulPartitionedCall?layer_1/StatefulPartitionedCall?layer_2/StatefulPartitionedCall?layer_3/StatefulPartitionedCall?
lambda/PartitionedCallPartitionedCalllambda_input*
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
GPU2*0J 8? *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_1966162
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196618layer_0_196620layer_0_196622layer_0_196624layer_0_196626layer_0_196628layer_0_196630layer_0_196632layer_0_196634layer_0_196636layer_0_196638*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962332!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196641layer_1_196643layer_1_196645layer_1_196647layer_1_196649layer_1_196651layer_1_196653layer_1_196655layer_1_196657layer_1_196659layer_1_196661*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1963972!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196664layer_2_196666layer_2_196668layer_2_196670layer_2_196672layer_2_196674layer_2_196676layer_2_196678layer_2_196680layer_2_196682layer_2_196684*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1965612!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196687layer_3_196689layer_3_196691*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_layer_3_layer_call_and_return_conditional_losses_1965992!
layer_3/StatefulPartitionedCall?
IdentityIdentity(layer_3/StatefulPartitionedCall:output:0 ^layer_0/StatefulPartitionedCall ^layer_1/StatefulPartitionedCall ^layer_2/StatefulPartitionedCall ^layer_3/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2B
layer_0/StatefulPartitionedCalllayer_0/StatefulPartitionedCall2B
layer_1/StatefulPartitionedCalllayer_1/StatefulPartitionedCall2B
layer_2/StatefulPartitionedCalllayer_2/StatefulPartitionedCall2B
layer_3/StatefulPartitionedCalllayer_3/StatefulPartitionedCall:o k
A
_output_shapes/
-:+???????????????????????????
&
_user_specified_namelambda_input:$ 

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
gdn_0_cond_1_cond_true_196129!
gdn_0_cond_1_cond_abs_biasadd!
gdn_0_cond_1_cond_placeholder
gdn_0_cond_1_cond_identity?
gdn_0/cond_1/cond/AbsAbsgdn_0_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Abs?
gdn_0/cond_1/cond/IdentityIdentitygdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity"A
gdn_0_cond_1_cond_identity#gdn_0/cond_1/cond/Identity:output:0*(
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
?
C__inference_encoder_layer_call_and_return_conditional_losses_198403

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	??
0analysis_layer_0_biasadd_readvariableop_resource:	?"
analysis_layer_0_gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y$
 analysis_layer_0_gdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
???
0analysis_layer_1_biasadd_readvariableop_resource:	?"
analysis_layer_1_gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y$
 analysis_layer_1_gdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
???
0analysis_layer_2_biasadd_readvariableop_resource:	?"
analysis_layer_2_gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y$
 analysis_layer_2_gdn_2_equal_1_x
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
???
0analysis_layer_3_biasadd_readvariableop_resource:	?
identity??'analysis/layer_0/BiasAdd/ReadVariableOp?'analysis/layer_1/BiasAdd/ReadVariableOp?'analysis/layer_2/BiasAdd/ReadVariableOp?'analysis/layer_3/BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp{
analysis/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
analysis/lambda/truediv/y?
analysis/lambda/truedivRealDivinputs"analysis/lambda/truediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
analysis/lambda/truediv?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
analysis/layer_0/Conv2DConv2Danalysis/lambda/truediv:z:0layer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_0/Conv2D?
'analysis/layer_0/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_0/BiasAdd/ReadVariableOp?
analysis/layer_0/BiasAddBiasAdd analysis/layer_0/Conv2D:output:0/analysis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_0/BiasAddy
analysis/layer_0/gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_0/gdn_0/x?
analysis/layer_0/gdn_0/EqualEqualanalysis_layer_0_gdn_0_equal_x!analysis/layer_0/gdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_0/gdn_0/Equal?
analysis/layer_0/gdn_0/condStatelessIf analysis/layer_0/gdn_0/Equal:z:0 analysis/layer_0/gdn_0/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_0_gdn_0_cond_false_197998*
output_shapes
: *:
then_branch+R)
'analysis_layer_0_gdn_0_cond_true_1979972
analysis/layer_0/gdn_0/cond?
$analysis/layer_0/gdn_0/cond/IdentityIdentity$analysis/layer_0/gdn_0/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity?
analysis/layer_0/gdn_0/cond_1StatelessIf-analysis/layer_0/gdn_0/cond/Identity:output:0!analysis/layer_0/BiasAdd:output:0analysis_layer_0_gdn_0_equal_x*
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
 *=
else_branch.R,
*analysis_layer_0_gdn_0_cond_1_false_198009*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_1_true_1980082
analysis/layer_0/gdn_0/cond_1?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity&analysis/layer_0/gdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198054*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198064*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
$analysis/layer_0/gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_0/gdn_0/Reshape/shape?
analysis/layer_0/gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0-analysis/layer_0/gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_0/gdn_0/Reshape?
"analysis/layer_0/gdn_0/convolutionConv2D/analysis/layer_0/gdn_0/cond_1/Identity:output:0'analysis/layer_0/gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_0/gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198078*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
analysis/layer_0/gdn_0/BiasAddBiasAdd+analysis/layer_0/gdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_0/gdn_0/BiasAdd}
analysis/layer_0/gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_0/gdn_0/x_1?
analysis/layer_0/gdn_0/Equal_1Equal analysis_layer_0_gdn_0_equal_1_x#analysis/layer_0/gdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_0/gdn_0/Equal_1?
analysis/layer_0/gdn_0/cond_2StatelessIf"analysis/layer_0/gdn_0/Equal_1:z:0'analysis/layer_0/gdn_0/BiasAdd:output:0 analysis_layer_0_gdn_0_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_0_gdn_0_cond_2_false_198092*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_2_true_1980912
analysis/layer_0/gdn_0/cond_2?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentity&analysis/layer_0/gdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity?
analysis/layer_0/gdn_0/truedivRealDiv!analysis/layer_0/BiasAdd:output:0/analysis/layer_0/gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_0/gdn_0/truediv?
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
layer_1/kernel/Reshape?
analysis/layer_1/Conv2DConv2D"analysis/layer_0/gdn_0/truediv:z:0layer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_1/Conv2D?
'analysis/layer_1/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_1/BiasAdd/ReadVariableOp?
analysis/layer_1/BiasAddBiasAdd analysis/layer_1/Conv2D:output:0/analysis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_1/BiasAddy
analysis/layer_1/gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_1/gdn_1/x?
analysis/layer_1/gdn_1/EqualEqualanalysis_layer_1_gdn_1_equal_x!analysis/layer_1/gdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_1/gdn_1/Equal?
analysis/layer_1/gdn_1/condStatelessIf analysis/layer_1/gdn_1/Equal:z:0 analysis/layer_1/gdn_1/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_1_gdn_1_cond_false_198134*
output_shapes
: *:
then_branch+R)
'analysis_layer_1_gdn_1_cond_true_1981332
analysis/layer_1/gdn_1/cond?
$analysis/layer_1/gdn_1/cond/IdentityIdentity$analysis/layer_1/gdn_1/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity?
analysis/layer_1/gdn_1/cond_1StatelessIf-analysis/layer_1/gdn_1/cond/Identity:output:0!analysis/layer_1/BiasAdd:output:0analysis_layer_1_gdn_1_equal_x*
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
 *=
else_branch.R,
*analysis_layer_1_gdn_1_cond_1_false_198145*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_1_true_1981442
analysis/layer_1/gdn_1/cond_1?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity&analysis/layer_1/gdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198190*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198200*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
$analysis/layer_1/gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_1/gdn_1/Reshape/shape?
analysis/layer_1/gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0-analysis/layer_1/gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_1/gdn_1/Reshape?
"analysis/layer_1/gdn_1/convolutionConv2D/analysis/layer_1/gdn_1/cond_1/Identity:output:0'analysis/layer_1/gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_1/gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198214*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
analysis/layer_1/gdn_1/BiasAddBiasAdd+analysis/layer_1/gdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_1/gdn_1/BiasAdd}
analysis/layer_1/gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_1/gdn_1/x_1?
analysis/layer_1/gdn_1/Equal_1Equal analysis_layer_1_gdn_1_equal_1_x#analysis/layer_1/gdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_1/gdn_1/Equal_1?
analysis/layer_1/gdn_1/cond_2StatelessIf"analysis/layer_1/gdn_1/Equal_1:z:0'analysis/layer_1/gdn_1/BiasAdd:output:0 analysis_layer_1_gdn_1_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_1_gdn_1_cond_2_false_198228*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_2_true_1982272
analysis/layer_1/gdn_1/cond_2?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentity&analysis/layer_1/gdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity?
analysis/layer_1/gdn_1/truedivRealDiv!analysis/layer_1/BiasAdd:output:0/analysis/layer_1/gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_1/gdn_1/truediv?
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
layer_2/kernel/Reshape?
analysis/layer_2/Conv2DConv2D"analysis/layer_1/gdn_1/truediv:z:0layer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_2/Conv2D?
'analysis/layer_2/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_2/BiasAdd/ReadVariableOp?
analysis/layer_2/BiasAddBiasAdd analysis/layer_2/Conv2D:output:0/analysis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_2/BiasAddy
analysis/layer_2/gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_2/gdn_2/x?
analysis/layer_2/gdn_2/EqualEqualanalysis_layer_2_gdn_2_equal_x!analysis/layer_2/gdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_2/gdn_2/Equal?
analysis/layer_2/gdn_2/condStatelessIf analysis/layer_2/gdn_2/Equal:z:0 analysis/layer_2/gdn_2/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_2_gdn_2_cond_false_198270*
output_shapes
: *:
then_branch+R)
'analysis_layer_2_gdn_2_cond_true_1982692
analysis/layer_2/gdn_2/cond?
$analysis/layer_2/gdn_2/cond/IdentityIdentity$analysis/layer_2/gdn_2/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity?
analysis/layer_2/gdn_2/cond_1StatelessIf-analysis/layer_2/gdn_2/cond/Identity:output:0!analysis/layer_2/BiasAdd:output:0analysis_layer_2_gdn_2_equal_x*
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
 *=
else_branch.R,
*analysis_layer_2_gdn_2_cond_1_false_198281*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_1_true_1982802
analysis/layer_2/gdn_2/cond_1?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity&analysis/layer_2/gdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198326*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198336*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
$analysis/layer_2/gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_2/gdn_2/Reshape/shape?
analysis/layer_2/gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0-analysis/layer_2/gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_2/gdn_2/Reshape?
"analysis/layer_2/gdn_2/convolutionConv2D/analysis/layer_2/gdn_2/cond_1/Identity:output:0'analysis/layer_2/gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_2/gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198350*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
analysis/layer_2/gdn_2/BiasAddBiasAdd+analysis/layer_2/gdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_2/gdn_2/BiasAdd}
analysis/layer_2/gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_2/gdn_2/x_1?
analysis/layer_2/gdn_2/Equal_1Equal analysis_layer_2_gdn_2_equal_1_x#analysis/layer_2/gdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_2/gdn_2/Equal_1?
analysis/layer_2/gdn_2/cond_2StatelessIf"analysis/layer_2/gdn_2/Equal_1:z:0'analysis/layer_2/gdn_2/BiasAdd:output:0 analysis_layer_2_gdn_2_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_2_gdn_2_cond_2_false_198364*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_2_true_1983632
analysis/layer_2/gdn_2/cond_2?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentity&analysis/layer_2/gdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity?
analysis/layer_2/gdn_2/truedivRealDiv!analysis/layer_2/BiasAdd:output:0/analysis/layer_2/gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_2/gdn_2/truediv?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
analysis/layer_3/Conv2DConv2D"analysis/layer_2/gdn_2/truediv:z:0layer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_3/Conv2D?
'analysis/layer_3/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_3/BiasAdd/ReadVariableOp?
analysis/layer_3/BiasAddBiasAdd analysis/layer_3/Conv2D:output:0/analysis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_3/BiasAdd?
IdentityIdentity!analysis/layer_3/BiasAdd:output:0(^analysis/layer_0/BiasAdd/ReadVariableOp(^analysis/layer_1/BiasAdd/ReadVariableOp(^analysis/layer_2/BiasAdd/ReadVariableOp(^analysis/layer_3/BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2R
'analysis/layer_0/BiasAdd/ReadVariableOp'analysis/layer_0/BiasAdd/ReadVariableOp2R
'analysis/layer_1/BiasAdd/ReadVariableOp'analysis/layer_1/BiasAdd/ReadVariableOp2R
'analysis/layer_2/BiasAdd/ReadVariableOp'analysis/layer_2/BiasAdd/ReadVariableOp2R
'analysis/layer_3/BiasAdd/ReadVariableOp'analysis/layer_3/BiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
/analysis_layer_0_gdn_0_cond_1_cond_false_197594D
@analysis_layer_0_gdn_0_cond_1_cond_cond_analysis_layer_0_biasadd.
*analysis_layer_0_gdn_0_cond_1_cond_equal_x/
+analysis_layer_0_gdn_0_cond_1_cond_identity?
$analysis/layer_0/gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_0/gdn_0/cond_1/cond/x?
(analysis/layer_0/gdn_0/cond_1/cond/EqualEqual*analysis_layer_0_gdn_0_cond_1_cond_equal_x-analysis/layer_0/gdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_0/gdn_0/cond_1/cond/Equal?
'analysis/layer_0/gdn_0/cond_1/cond/condStatelessIf,analysis/layer_0/gdn_0/cond_1/cond/Equal:z:0@analysis_layer_0_gdn_0_cond_1_cond_cond_analysis_layer_0_biasadd*analysis_layer_0_gdn_0_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_197604*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_1976032)
'analysis/layer_0/gdn_0/cond_1/cond/cond?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity0analysis/layer_0/gdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity"c
+analysis_layer_0_gdn_0_cond_1_cond_identity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
w
layer_2_gdn_2_cond_false_1991183
/layer_2_gdn_2_cond_identity_layer_2_gdn_2_equal

layer_2_gdn_2_cond_identity
?
layer_2/gdn_2/cond/IdentityIdentity/layer_2_gdn_2_cond_identity_layer_2_gdn_2_equal*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity"C
layer_2_gdn_2_cond_identity$layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?	
?
<encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_false_195978X
Tencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_pow_encoder_analysis_layer_2_biasadd9
5encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y<
8encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
3encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/powPowTencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_pow_encoder_analysis_layer_2_biasadd5encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/pow?
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity7encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_identityAencoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
u
gdn_2_cond_2_false_199666#
gdn_2_cond_2_cond_gdn_2_biasadd
gdn_2_cond_2_equal_x
gdn_2_cond_2_identitye
gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_2/cond_2/x?
gdn_2/cond_2/EqualEqualgdn_2_cond_2_equal_xgdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_2/Equal?
gdn_2/cond_2/condStatelessIfgdn_2/cond_2/Equal:z:0gdn_2_cond_2_cond_gdn_2_biasaddgdn_2_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_2_cond_2_cond_false_199675*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_2_cond_true_1996742
gdn_2/cond_2/cond?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity?
gdn_2/cond_2/IdentityIdentity#gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity"7
gdn_2_cond_2_identitygdn_2/cond_2/Identity:output:0*(
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
"gdn_0_cond_1_cond_cond_true_199313)
%gdn_0_cond_1_cond_cond_square_biasadd&
"gdn_0_cond_1_cond_cond_placeholder#
gdn_0_cond_1_cond_cond_identity?
gdn_0/cond_1/cond/cond/SquareSquare%gdn_0_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/cond/Square?
gdn_0/cond_1/cond/cond/IdentityIdentity!gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity"K
gdn_0_cond_1_cond_cond_identity(gdn_0/cond_1/cond/cond/Identity:output:0*(
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
*analysis_layer_0_gdn_0_cond_2_false_197668E
Aanalysis_layer_0_gdn_0_cond_2_cond_analysis_layer_0_gdn_0_biasadd)
%analysis_layer_0_gdn_0_cond_2_equal_x*
&analysis_layer_0_gdn_0_cond_2_identity?
analysis/layer_0/gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_0/gdn_0/cond_2/x?
#analysis/layer_0/gdn_0/cond_2/EqualEqual%analysis_layer_0_gdn_0_cond_2_equal_x(analysis/layer_0/gdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_0/gdn_0/cond_2/Equal?
"analysis/layer_0/gdn_0/cond_2/condStatelessIf'analysis/layer_0/gdn_0/cond_2/Equal:z:0Aanalysis_layer_0_gdn_0_cond_2_cond_analysis_layer_0_gdn_0_biasadd%analysis_layer_0_gdn_0_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_0_gdn_0_cond_2_cond_false_197677*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_2_cond_true_1976762$
"analysis/layer_0/gdn_0/cond_2/cond?
+analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_2/cond/Identity?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentity4analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity"Y
&analysis_layer_0_gdn_0_cond_2_identity/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
 layer_1_gdn_1_cond_2_true_1986517
3layer_1_gdn_1_cond_2_identity_layer_1_gdn_1_biasadd$
 layer_1_gdn_1_cond_2_placeholder!
layer_1_gdn_1_cond_2_identity?
layer_1/gdn_1/cond_2/IdentityIdentity3layer_1_gdn_1_cond_2_identity_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity"G
layer_1_gdn_1_cond_2_identity&layer_1/gdn_1/cond_2/Identity:output:0*(
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
(__inference_encoder_layer_call_fn_197322
input_1
unknown
	unknown_0:	?
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

unknown_33:
??

unknown_34:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1972472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_197876H
Danalysis_layer_2_gdn_2_cond_1_cond_cond_pow_analysis_layer_2_biasadd1
-analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y4
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
+analysis/layer_2/gdn_2/cond_1/cond/cond/powPowDanalysis_layer_2_gdn_2_cond_1_cond_cond_pow_analysis_layer_2_biasadd-analysis_layer_2_gdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/cond/pow?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity/analysis/layer_2/gdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"m
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
.analysis_layer_1_gdn_1_cond_2_cond_true_197812J
Fanalysis_layer_1_gdn_1_cond_2_cond_sqrt_analysis_layer_1_gdn_1_biasadd2
.analysis_layer_1_gdn_1_cond_2_cond_placeholder/
+analysis_layer_1_gdn_1_cond_2_cond_identity?
'analysis/layer_1/gdn_1/cond_2/cond/SqrtSqrtFanalysis_layer_1_gdn_1_cond_2_cond_sqrt_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_1/gdn_1/cond_2/cond/Sqrt?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity"c
+analysis_layer_1_gdn_1_cond_2_cond_identity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
?
7encoder_analysis_layer_0_gdn_0_cond_1_cond_false_195696T
Pencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_encoder_analysis_layer_0_biasadd6
2encoder_analysis_layer_0_gdn_0_cond_1_cond_equal_x7
3encoder_analysis_layer_0_gdn_0_cond_1_cond_identity?
,encoder/analysis/layer_0/gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,encoder/analysis/layer_0/gdn_0/cond_1/cond/x?
0encoder/analysis/layer_0/gdn_0/cond_1/cond/EqualEqual2encoder_analysis_layer_0_gdn_0_cond_1_cond_equal_x5encoder/analysis/layer_0/gdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0encoder/analysis/layer_0/gdn_0/cond_1/cond/Equal?
/encoder/analysis/layer_0/gdn_0/cond_1/cond/condStatelessIf4encoder/analysis/layer_0/gdn_0/cond_1/cond/Equal:z:0Pencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_encoder_analysis_layer_0_biasadd2encoder_analysis_layer_0_gdn_0_cond_1_cond_equal_x*
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
<encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_false_195706*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_true_19570521
/encoder/analysis/layer_0/gdn_0/cond_1/cond/cond?
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity?
3encoder/analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentityAencoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity"s
3encoder_analysis_layer_0_gdn_0_cond_1_cond_identity<encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
u
gdn_0_cond_2_false_196204#
gdn_0_cond_2_cond_gdn_0_biasadd
gdn_0_cond_2_equal_x
gdn_0_cond_2_identitye
gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_0/cond_2/x?
gdn_0/cond_2/EqualEqualgdn_0_cond_2_equal_xgdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_2/Equal?
gdn_0/cond_2/condStatelessIfgdn_0/cond_2/Equal:z:0gdn_0_cond_2_cond_gdn_0_biasaddgdn_0_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_0_cond_2_cond_false_196213*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_2_cond_true_1962122
gdn_0/cond_2/cond?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity?
gdn_0/cond_2/IdentityIdentity#gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity"7
gdn_0_cond_2_identitygdn_0/cond_2/Identity:output:0*(
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
u
gdn_0_cond_2_false_199378#
gdn_0_cond_2_cond_gdn_0_biasadd
gdn_0_cond_2_equal_x
gdn_0_cond_2_identitye
gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_0/cond_2/x?
gdn_0/cond_2/EqualEqualgdn_0_cond_2_equal_xgdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_2/Equal?
gdn_0/cond_2/condStatelessIfgdn_0/cond_2/Equal:z:0gdn_0_cond_2_cond_gdn_0_biasaddgdn_0_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_0_cond_2_cond_false_199387*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_2_cond_true_1993862
gdn_0/cond_2/cond?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity?
gdn_0/cond_2/IdentityIdentity#gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity"7
gdn_0_cond_2_identitygdn_0/cond_2/Identity:output:0*(
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
o
gdn_1_cond_1_false_196285
gdn_1_cond_1_cond_biasadd
gdn_1_cond_1_equal_x
gdn_1_cond_1_identitye
gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_1/cond_1/x?
gdn_1/cond_1/EqualEqualgdn_1_cond_1_equal_xgdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_1/Equal?
gdn_1/cond_1/condStatelessIfgdn_1/cond_1/Equal:z:0gdn_1_cond_1_cond_biasaddgdn_1_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_1_cond_1_cond_false_196294*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_1_cond_true_1962932
gdn_1/cond_1/cond?
gdn_1/cond_1/cond/IdentityIdentitygdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity?
gdn_1/cond_1/IdentityIdentity#gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity"7
gdn_1_cond_1_identitygdn_1/cond_1/Identity:output:0*(
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
|
gdn_0_cond_2_true_196203'
#gdn_0_cond_2_identity_gdn_0_biasadd
gdn_0_cond_2_placeholder
gdn_0_cond_2_identity?
gdn_0/cond_2/IdentityIdentity#gdn_0_cond_2_identity_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity"7
gdn_0_cond_2_identitygdn_0/cond_2/Identity:output:0*(
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
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_197875K
Ganalysis_layer_2_gdn_2_cond_1_cond_cond_square_analysis_layer_2_biasadd7
3analysis_layer_2_gdn_2_cond_1_cond_cond_placeholder4
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
.analysis/layer_2/gdn_2/cond_1/cond/cond/SquareSquareGanalysis_layer_2_gdn_2_cond_1_cond_cond_square_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_2/gdn_2/cond_1/cond/cond/Square?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity2analysis/layer_2/gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"m
0analysis_layer_2_gdn_2_cond_1_cond_cond_identity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
(analysis_layer_1_gdn_1_cond_false_198134E
Aanalysis_layer_1_gdn_1_cond_identity_analysis_layer_1_gdn_1_equal
(
$analysis_layer_1_gdn_1_cond_identity
?
$analysis/layer_1/gdn_1/cond/IdentityIdentityAanalysis_layer_1_gdn_1_cond_identity_analysis_layer_1_gdn_1_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity"U
$analysis_layer_1_gdn_1_cond_identity-analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
gdn_0_cond_2_cond_true_196212(
$gdn_0_cond_2_cond_sqrt_gdn_0_biasadd!
gdn_0_cond_2_cond_placeholder
gdn_0_cond_2_cond_identity?
gdn_0/cond_2/cond/SqrtSqrt$gdn_0_cond_2_cond_sqrt_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Sqrt?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity"A
gdn_0_cond_2_cond_identity#gdn_0/cond_2/cond/Identity:output:0*(
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
(analysis_layer_2_gdn_2_cond_false_197846E
Aanalysis_layer_2_gdn_2_cond_identity_analysis_layer_2_gdn_2_equal
(
$analysis_layer_2_gdn_2_cond_identity
?
$analysis/layer_2/gdn_2/cond/IdentityIdentityAanalysis_layer_2_gdn_2_cond_identity_analysis_layer_2_gdn_2_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity"U
$analysis_layer_2_gdn_2_cond_identity-analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?N
?
C__inference_layer_0_layer_call_and_return_conditional_losses_199407

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y
gdn_0_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
Conv2DConv2Dinputslayer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_0/x?
gdn_0/EqualEqualgdn_0_equal_xgdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/Equal?

gdn_0/condStatelessIfgdn_0/Equal:z:0gdn_0/Equal:z:0*
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
 **
else_branchR
gdn_0_cond_false_199284*
output_shapes
: *)
then_branchR
gdn_0_cond_true_1992832

gdn_0/condl
gdn_0/cond/IdentityIdentitygdn_0/cond:output:0*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity?
gdn_0/cond_1StatelessIfgdn_0/cond/Identity:output:0BiasAdd:output:0gdn_0_equal_x*
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
 *,
else_branchR
gdn_0_cond_1_false_199295*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_1_true_1992942
gdn_0/cond_1?
gdn_0/cond_1/IdentityIdentitygdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199340*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199350*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_0/Reshape/shape?
gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_0/Reshape?
gdn_0/convolutionConv2Dgdn_0/cond_1/Identity:output:0gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199364*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
gdn_0/BiasAddBiasAddgdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/BiasAdd[
	gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_0/x_1?
gdn_0/Equal_1Equalgdn_0_equal_1_xgdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/Equal_1?
gdn_0/cond_2StatelessIfgdn_0/Equal_1:z:0gdn_0/BiasAdd:output:0gdn_0_equal_1_x*
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
 *,
else_branchR
gdn_0_cond_2_false_199378*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_2_true_1993772
gdn_0/cond_2?
gdn_0/cond_2/IdentityIdentitygdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity?
gdn_0/truedivRealDivBiasAdd:output:0gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/truediv?
IdentityIdentitygdn_0/truediv:z:0^BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:+???????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
?
?
&layer_2_gdn_2_cond_2_cond_false_1987977
3layer_2_gdn_2_cond_2_cond_pow_layer_2_gdn_2_biasadd#
layer_2_gdn_2_cond_2_cond_pow_y&
"layer_2_gdn_2_cond_2_cond_identity?
layer_2/gdn_2/cond_2/cond/powPow3layer_2_gdn_2_cond_2_cond_pow_layer_2_gdn_2_biasaddlayer_2_gdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/cond/pow?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity!layer_2/gdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity"Q
"layer_2_gdn_2_cond_2_cond_identity+layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
+layer_1_gdn_1_cond_1_cond_cond_false_1990126
2layer_1_gdn_1_cond_1_cond_cond_pow_layer_1_biasadd(
$layer_1_gdn_1_cond_1_cond_cond_pow_y+
'layer_1_gdn_1_cond_1_cond_cond_identity?
"layer_1/gdn_1/cond_1/cond/cond/powPow2layer_1_gdn_1_cond_1_cond_cond_pow_layer_1_biasadd$layer_1_gdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/cond/pow?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity&layer_1/gdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity"[
'layer_1_gdn_1_cond_1_cond_cond_identity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
W
gdn_1_cond_false_196274#
gdn_1_cond_identity_gdn_1_equal

gdn_1_cond_identity
x
gdn_1/cond/IdentityIdentitygdn_1_cond_identity_gdn_1_equal*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity"3
gdn_1_cond_identitygdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
W
gdn_1_cond_false_199428#
gdn_1_cond_identity_gdn_1_equal

gdn_1_cond_identity
x
gdn_1/cond/IdentityIdentitygdn_1_cond_identity_gdn_1_equal*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity"3
gdn_1_cond_identitygdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
*analysis_layer_2_gdn_2_cond_1_false_198281?
;analysis_layer_2_gdn_2_cond_1_cond_analysis_layer_2_biasadd)
%analysis_layer_2_gdn_2_cond_1_equal_x*
&analysis_layer_2_gdn_2_cond_1_identity?
analysis/layer_2/gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_2/gdn_2/cond_1/x?
#analysis/layer_2/gdn_2/cond_1/EqualEqual%analysis_layer_2_gdn_2_cond_1_equal_x(analysis/layer_2/gdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_2/gdn_2/cond_1/Equal?
"analysis/layer_2/gdn_2/cond_1/condStatelessIf'analysis/layer_2/gdn_2/cond_1/Equal:z:0;analysis_layer_2_gdn_2_cond_1_cond_analysis_layer_2_biasadd%analysis_layer_2_gdn_2_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_2_gdn_2_cond_1_cond_false_198290*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_1_cond_true_1982892$
"analysis/layer_2/gdn_2/cond_1/cond?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity"Y
&analysis_layer_2_gdn_2_cond_1_identity/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
+layer_0_gdn_0_cond_1_cond_cond_false_1988766
2layer_0_gdn_0_cond_1_cond_cond_pow_layer_0_biasadd(
$layer_0_gdn_0_cond_1_cond_cond_pow_y+
'layer_0_gdn_0_cond_1_cond_cond_identity?
"layer_0/gdn_0/cond_1/cond/cond/powPow2layer_0_gdn_0_cond_1_cond_cond_pow_layer_0_biasadd$layer_0_gdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/cond/pow?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity&layer_0/gdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity"[
'layer_0_gdn_0_cond_1_cond_cond_identity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
2encoder_analysis_layer_1_gdn_1_cond_1_false_195823O
Kencoder_analysis_layer_1_gdn_1_cond_1_cond_encoder_analysis_layer_1_biasadd1
-encoder_analysis_layer_1_gdn_1_cond_1_equal_x2
.encoder_analysis_layer_1_gdn_1_cond_1_identity?
'encoder/analysis/layer_1/gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'encoder/analysis/layer_1/gdn_1/cond_1/x?
+encoder/analysis/layer_1/gdn_1/cond_1/EqualEqual-encoder_analysis_layer_1_gdn_1_cond_1_equal_x0encoder/analysis/layer_1/gdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_1/gdn_1/cond_1/Equal?
*encoder/analysis/layer_1/gdn_1/cond_1/condStatelessIf/encoder/analysis/layer_1/gdn_1/cond_1/Equal:z:0Kencoder_analysis_layer_1_gdn_1_cond_1_cond_encoder_analysis_layer_1_biasadd-encoder_analysis_layer_1_gdn_1_cond_1_equal_x*
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
7encoder_analysis_layer_1_gdn_1_cond_1_cond_false_195832*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_1_gdn_1_cond_1_cond_true_1958312,
*encoder/analysis/layer_1/gdn_1/cond_1/cond?
3encoder/analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity3encoder/analysis/layer_1/gdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity?
.encoder/analysis/layer_1/gdn_1/cond_1/IdentityIdentity<encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_1/Identity"i
.encoder_analysis_layer_1_gdn_1_cond_1_identity7encoder/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
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
!layer_2_gdn_2_cond_2_false_1992123
/layer_2_gdn_2_cond_2_cond_layer_2_gdn_2_biasadd 
layer_2_gdn_2_cond_2_equal_x!
layer_2_gdn_2_cond_2_identityu
layer_2/gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_2/gdn_2/cond_2/x?
layer_2/gdn_2/cond_2/EqualEquallayer_2_gdn_2_cond_2_equal_xlayer_2/gdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/cond_2/Equal?
layer_2/gdn_2/cond_2/condStatelessIflayer_2/gdn_2/cond_2/Equal:z:0/layer_2_gdn_2_cond_2_cond_layer_2_gdn_2_biasaddlayer_2_gdn_2_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_2_gdn_2_cond_2_cond_false_199221*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_2_cond_true_1992202
layer_2/gdn_2/cond_2/cond?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity"layer_2/gdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity?
layer_2/gdn_2/cond_2/IdentityIdentity+layer_2/gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity"G
layer_2_gdn_2_cond_2_identity&layer_2/gdn_2/cond_2/Identity:output:0*(
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
)analysis_layer_2_gdn_2_cond_1_true_197856C
?analysis_layer_2_gdn_2_cond_1_identity_analysis_layer_2_biasadd-
)analysis_layer_2_gdn_2_cond_1_placeholder*
&analysis_layer_2_gdn_2_cond_1_identity?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity?analysis_layer_2_gdn_2_cond_1_identity_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity"Y
&analysis_layer_2_gdn_2_cond_1_identity/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
!layer_0_gdn_0_cond_1_false_198857-
)layer_0_gdn_0_cond_1_cond_layer_0_biasadd 
layer_0_gdn_0_cond_1_equal_x!
layer_0_gdn_0_cond_1_identityu
layer_0/gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/cond_1/x?
layer_0/gdn_0/cond_1/EqualEquallayer_0_gdn_0_cond_1_equal_xlayer_0/gdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/cond_1/Equal?
layer_0/gdn_0/cond_1/condStatelessIflayer_0/gdn_0/cond_1/Equal:z:0)layer_0_gdn_0_cond_1_cond_layer_0_biasaddlayer_0_gdn_0_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_0_gdn_0_cond_1_cond_false_198866*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_1_cond_true_1988652
layer_0/gdn_0/cond_1/cond?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity"layer_0/gdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity?
layer_0/gdn_0/cond_1/IdentityIdentity+layer_0/gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity"G
layer_0_gdn_0_cond_1_identity&layer_0/gdn_0/cond_1/Identity:output:0*(
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
&layer_2_gdn_2_cond_2_cond_false_1992217
3layer_2_gdn_2_cond_2_cond_pow_layer_2_gdn_2_biasadd#
layer_2_gdn_2_cond_2_cond_pow_y&
"layer_2_gdn_2_cond_2_cond_identity?
layer_2/gdn_2/cond_2/cond/powPow3layer_2_gdn_2_cond_2_cond_pow_layer_2_gdn_2_biasaddlayer_2_gdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/cond/pow?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity!layer_2/gdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity"Q
"layer_2_gdn_2_cond_2_cond_identity+layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
/encoder_analysis_layer_0_gdn_0_cond_true_1956753
/encoder_analysis_layer_0_gdn_0_cond_placeholder
0
,encoder_analysis_layer_0_gdn_0_cond_identity
?
)encoder/analysis/layer_0/gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)encoder/analysis/layer_0/gdn_0/cond/Const?
,encoder/analysis/layer_0/gdn_0/cond/IdentityIdentity2encoder/analysis/layer_0/gdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_0/gdn_0/cond/Identity"e
,encoder_analysis_layer_0_gdn_0_cond_identity5encoder/analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?N
?
C__inference_layer_1_layer_call_and_return_conditional_losses_199551

inputs
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y
gdn_1_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?
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
Conv2DConv2Dinputslayer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_1/x?
gdn_1/EqualEqualgdn_1_equal_xgdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/Equal?

gdn_1/condStatelessIfgdn_1/Equal:z:0gdn_1/Equal:z:0*
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
 **
else_branchR
gdn_1_cond_false_199428*
output_shapes
: *)
then_branchR
gdn_1_cond_true_1994272

gdn_1/condl
gdn_1/cond/IdentityIdentitygdn_1/cond:output:0*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity?
gdn_1/cond_1StatelessIfgdn_1/cond/Identity:output:0BiasAdd:output:0gdn_1_equal_x*
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
 *,
else_branchR
gdn_1_cond_1_false_199439*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_1_true_1994382
gdn_1/cond_1?
gdn_1/cond_1/IdentityIdentitygdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199484*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199494*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_1/Reshape/shape?
gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_1/Reshape?
gdn_1/convolutionConv2Dgdn_1/cond_1/Identity:output:0gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199508*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
gdn_1/BiasAddBiasAddgdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/BiasAdd[
	gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_1/x_1?
gdn_1/Equal_1Equalgdn_1_equal_1_xgdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/Equal_1?
gdn_1/cond_2StatelessIfgdn_1/Equal_1:z:0gdn_1/BiasAdd:output:0gdn_1_equal_1_x*
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
 *,
else_branchR
gdn_1_cond_2_false_199522*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_2_true_1995212
gdn_1/cond_2?
gdn_1/cond_2/IdentityIdentitygdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity?
gdn_1/truedivRealDivBiasAdd:output:0gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/truediv?
IdentityIdentitygdn_1/truediv:z:0^BiasAdd/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
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
?
W
gdn_2_cond_false_199572#
gdn_2_cond_identity_gdn_2_equal

gdn_2_cond_identity
x
gdn_2/cond/IdentityIdentitygdn_2_cond_identity_gdn_2_equal*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity"3
gdn_2_cond_identitygdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?N
?
C__inference_layer_2_layer_call_and_return_conditional_losses_196561

inputs
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y
gdn_2_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?
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
Conv2DConv2Dinputslayer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_2/x?
gdn_2/EqualEqualgdn_2_equal_xgdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/Equal?

gdn_2/condStatelessIfgdn_2/Equal:z:0gdn_2/Equal:z:0*
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
 **
else_branchR
gdn_2_cond_false_196438*
output_shapes
: *)
then_branchR
gdn_2_cond_true_1964372

gdn_2/condl
gdn_2/cond/IdentityIdentitygdn_2/cond:output:0*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity?
gdn_2/cond_1StatelessIfgdn_2/cond/Identity:output:0BiasAdd:output:0gdn_2_equal_x*
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
 *,
else_branchR
gdn_2_cond_1_false_196449*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_1_true_1964482
gdn_2/cond_1?
gdn_2/cond_1/IdentityIdentitygdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196494*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196504*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_2/Reshape/shape?
gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_2/Reshape?
gdn_2/convolutionConv2Dgdn_2/cond_1/Identity:output:0gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196518*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
gdn_2/BiasAddBiasAddgdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/BiasAdd[
	gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_2/x_1?
gdn_2/Equal_1Equalgdn_2_equal_1_xgdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/Equal_1?
gdn_2/cond_2StatelessIfgdn_2/Equal_1:z:0gdn_2/BiasAdd:output:0gdn_2_equal_1_x*
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
 *,
else_branchR
gdn_2_cond_2_false_196532*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_2_true_1965312
gdn_2/cond_2?
gdn_2/cond_2/IdentityIdentitygdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity?
gdn_2/truedivRealDivBiasAdd:output:0gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/truediv?
IdentityIdentitygdn_2/truediv:z:0^BiasAdd/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
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
0encoder_analysis_layer_1_gdn_1_cond_false_195812U
Qencoder_analysis_layer_1_gdn_1_cond_identity_encoder_analysis_layer_1_gdn_1_equal
0
,encoder_analysis_layer_1_gdn_1_cond_identity
?
,encoder/analysis/layer_1/gdn_1/cond/IdentityIdentityQencoder_analysis_layer_1_gdn_1_cond_identity_encoder_analysis_layer_1_gdn_1_equal*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_1/gdn_1/cond/Identity"e
,encoder_analysis_layer_1_gdn_1_cond_identity5encoder/analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
gdn_1_cond_1_cond_false_196294"
gdn_1_cond_1_cond_cond_biasadd
gdn_1_cond_1_cond_equal_x
gdn_1_cond_1_cond_identityo
gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_1/cond_1/cond/x?
gdn_1/cond_1/cond/EqualEqualgdn_1_cond_1_cond_equal_xgdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_1/cond/Equal?
gdn_1/cond_1/cond/condStatelessIfgdn_1/cond_1/cond/Equal:z:0gdn_1_cond_1_cond_cond_biasaddgdn_1_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_1_cond_1_cond_cond_false_196304*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_1_cond_1_cond_cond_true_1963032
gdn_1/cond_1/cond/cond?
gdn_1/cond_1/cond/cond/IdentityIdentitygdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity?
gdn_1/cond_1/cond/IdentityIdentity(gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity"A
gdn_1_cond_1_cond_identity#gdn_1/cond_1/cond/Identity:output:0*(
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
*analysis_layer_1_gdn_1_cond_1_false_198145?
;analysis_layer_1_gdn_1_cond_1_cond_analysis_layer_1_biasadd)
%analysis_layer_1_gdn_1_cond_1_equal_x*
&analysis_layer_1_gdn_1_cond_1_identity?
analysis/layer_1/gdn_1/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_1/gdn_1/cond_1/x?
#analysis/layer_1/gdn_1/cond_1/EqualEqual%analysis_layer_1_gdn_1_cond_1_equal_x(analysis/layer_1/gdn_1/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_1/gdn_1/cond_1/Equal?
"analysis/layer_1/gdn_1/cond_1/condStatelessIf'analysis/layer_1/gdn_1/cond_1/Equal:z:0;analysis_layer_1_gdn_1_cond_1_cond_analysis_layer_1_biasadd%analysis_layer_1_gdn_1_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_1_gdn_1_cond_1_cond_false_198154*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_1_cond_true_1981532$
"analysis/layer_1/gdn_1/cond_1/cond?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity"Y
&analysis_layer_1_gdn_1_cond_1_identity/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
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
 layer_1_gdn_1_cond_1_true_1985681
-layer_1_gdn_1_cond_1_identity_layer_1_biasadd$
 layer_1_gdn_1_cond_1_placeholder!
layer_1_gdn_1_cond_1_identity?
layer_1/gdn_1/cond_1/IdentityIdentity-layer_1_gdn_1_cond_1_identity_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity"G
layer_1_gdn_1_cond_1_identity&layer_1/gdn_1/cond_1/Identity:output:0*(
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
/analysis_layer_1_gdn_1_cond_1_cond_false_197730D
@analysis_layer_1_gdn_1_cond_1_cond_cond_analysis_layer_1_biasadd.
*analysis_layer_1_gdn_1_cond_1_cond_equal_x/
+analysis_layer_1_gdn_1_cond_1_cond_identity?
$analysis/layer_1/gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_1/gdn_1/cond_1/cond/x?
(analysis/layer_1/gdn_1/cond_1/cond/EqualEqual*analysis_layer_1_gdn_1_cond_1_cond_equal_x-analysis/layer_1/gdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_1/gdn_1/cond_1/cond/Equal?
'analysis/layer_1/gdn_1/cond_1/cond/condStatelessIf,analysis/layer_1/gdn_1/cond_1/cond/Equal:z:0@analysis_layer_1_gdn_1_cond_1_cond_cond_analysis_layer_1_biasadd*analysis_layer_1_gdn_1_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_197740*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_1977392)
'analysis/layer_1/gdn_1/cond_1/cond/cond?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity0analysis/layer_1/gdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity"c
+analysis_layer_1_gdn_1_cond_1_cond_identity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
 layer_2_gdn_2_cond_2_true_1992117
3layer_2_gdn_2_cond_2_identity_layer_2_gdn_2_biasadd$
 layer_2_gdn_2_cond_2_placeholder!
layer_2_gdn_2_cond_2_identity?
layer_2/gdn_2/cond_2/IdentityIdentity3layer_2_gdn_2_cond_2_identity_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity"G
layer_2_gdn_2_cond_2_identity&layer_2/gdn_2/cond_2/Identity:output:0*(
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
!layer_0_gdn_0_cond_2_false_1989403
/layer_0_gdn_0_cond_2_cond_layer_0_gdn_0_biasadd 
layer_0_gdn_0_cond_2_equal_x!
layer_0_gdn_0_cond_2_identityu
layer_0/gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_0/gdn_0/cond_2/x?
layer_0/gdn_0/cond_2/EqualEquallayer_0_gdn_0_cond_2_equal_xlayer_0/gdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/cond_2/Equal?
layer_0/gdn_0/cond_2/condStatelessIflayer_0/gdn_0/cond_2/Equal:z:0/layer_0_gdn_0_cond_2_cond_layer_0_gdn_0_biasaddlayer_0_gdn_0_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_0_gdn_0_cond_2_cond_false_198949*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_2_cond_true_1989482
layer_0/gdn_0/cond_2/cond?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity"layer_0/gdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity?
layer_0/gdn_0/cond_2/IdentityIdentity+layer_0/gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity"G
layer_0_gdn_0_cond_2_identity&layer_0/gdn_0/cond_2/Identity:output:0*(
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
e
layer_1_gdn_1_cond_true_198557"
layer_1_gdn_1_cond_placeholder

layer_1_gdn_1_cond_identity
v
layer_1/gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_1/gdn_1/cond/Const?
layer_1/gdn_1/cond/IdentityIdentity!layer_1/gdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity"C
layer_1_gdn_1_cond_identity$layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_198164H
Danalysis_layer_1_gdn_1_cond_1_cond_cond_pow_analysis_layer_1_biasadd1
-analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y4
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
+analysis/layer_1/gdn_1/cond_1/cond/cond/powPowDanalysis_layer_1_gdn_1_cond_1_cond_cond_pow_analysis_layer_1_biasadd-analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/cond/pow?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity/analysis/layer_1/gdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"m
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
"gdn_0_cond_1_cond_cond_true_196139)
%gdn_0_cond_1_cond_cond_square_biasadd&
"gdn_0_cond_1_cond_cond_placeholder#
gdn_0_cond_1_cond_cond_identity?
gdn_0/cond_1/cond/cond/SquareSquare%gdn_0_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/cond/Square?
gdn_0/cond_1/cond/cond/IdentityIdentity!gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity"K
gdn_0_cond_1_cond_cond_identity(gdn_0/cond_1/cond/cond/Identity:output:0*(
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
%layer_1_gdn_1_cond_1_cond_true_1990011
-layer_1_gdn_1_cond_1_cond_abs_layer_1_biasadd)
%layer_1_gdn_1_cond_1_cond_placeholder&
"layer_1_gdn_1_cond_1_cond_identity?
layer_1/gdn_1/cond_1/cond/AbsAbs-layer_1_gdn_1_cond_1_cond_abs_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/cond/Abs?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity!layer_1/gdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity"Q
"layer_1_gdn_1_cond_1_cond_identity+layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
.analysis_layer_2_gdn_2_cond_1_cond_true_197865C
?analysis_layer_2_gdn_2_cond_1_cond_abs_analysis_layer_2_biasadd2
.analysis_layer_2_gdn_2_cond_1_cond_placeholder/
+analysis_layer_2_gdn_2_cond_1_cond_identity?
&analysis/layer_2/gdn_2/cond_1/cond/AbsAbs?analysis_layer_2_gdn_2_cond_1_cond_abs_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/cond/Abs?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity"c
+analysis_layer_2_gdn_2_cond_1_cond_identity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?(
?
__inference__traced_save_199830
file_prefix+
'savev2_layer_0_bias_read_readvariableop9
5savev2_layer_0_gdn_0_reparam_beta_read_readvariableop:
6savev2_layer_0_gdn_0_reparam_gamma_read_readvariableop2
.savev2_layer_0_kernel_rdft_read_readvariableop+
'savev2_layer_1_bias_read_readvariableop9
5savev2_layer_1_gdn_1_reparam_beta_read_readvariableop:
6savev2_layer_1_gdn_1_reparam_gamma_read_readvariableop2
.savev2_layer_1_kernel_rdft_read_readvariableop+
'savev2_layer_2_bias_read_readvariableop9
5savev2_layer_2_gdn_2_reparam_beta_read_readvariableop:
6savev2_layer_2_gdn_2_reparam_gamma_read_readvariableop2
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
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_layer_0_bias_read_readvariableop5savev2_layer_0_gdn_0_reparam_beta_read_readvariableop6savev2_layer_0_gdn_0_reparam_gamma_read_readvariableop.savev2_layer_0_kernel_rdft_read_readvariableop'savev2_layer_1_bias_read_readvariableop5savev2_layer_1_gdn_1_reparam_beta_read_readvariableop6savev2_layer_1_gdn_1_reparam_gamma_read_readvariableop.savev2_layer_1_kernel_rdft_read_readvariableop'savev2_layer_2_bias_read_readvariableop5savev2_layer_2_gdn_2_reparam_beta_read_readvariableop6savev2_layer_2_gdn_2_reparam_gamma_read_readvariableop.savev2_layer_2_kernel_rdft_read_readvariableop'savev2_layer_3_bias_read_readvariableop.savev2_layer_3_kernel_rdft_read_readvariableopsavev2_const_22"/device:CPU:0*
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
??:	?:?:?:
??:
??:?:?:
??:
??:?:
??: 2(
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
??:%!

_output_shapes
:	?:!
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
??:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:

_output_shapes
: 
?
?
.analysis_layer_2_gdn_2_cond_2_cond_true_198372J
Fanalysis_layer_2_gdn_2_cond_2_cond_sqrt_analysis_layer_2_gdn_2_biasadd2
.analysis_layer_2_gdn_2_cond_2_cond_placeholder/
+analysis_layer_2_gdn_2_cond_2_cond_identity?
'analysis/layer_2/gdn_2/cond_2/cond/SqrtSqrtFanalysis_layer_2_gdn_2_cond_2_cond_sqrt_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_2/gdn_2/cond_2/cond/Sqrt?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity"c
+analysis_layer_2_gdn_2_cond_2_cond_identity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
'analysis_layer_2_gdn_2_cond_true_197845+
'analysis_layer_2_gdn_2_cond_placeholder
(
$analysis_layer_2_gdn_2_cond_identity
?
!analysis/layer_2/gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_2/gdn_2/cond/Const?
$analysis/layer_2/gdn_2/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity"U
$analysis_layer_2_gdn_2_cond_identity-analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
*layer_2_gdn_2_cond_1_cond_cond_true_1991479
5layer_2_gdn_2_cond_1_cond_cond_square_layer_2_biasadd.
*layer_2_gdn_2_cond_1_cond_cond_placeholder+
'layer_2_gdn_2_cond_1_cond_cond_identity?
%layer_2/gdn_2/cond_1/cond/cond/SquareSquare5layer_2_gdn_2_cond_1_cond_cond_square_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_2/gdn_2/cond_1/cond/cond/Square?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity)layer_2/gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity"[
'layer_2_gdn_2_cond_1_cond_cond_identity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
v
gdn_2_cond_1_true_199582!
gdn_2_cond_1_identity_biasadd
gdn_2_cond_1_placeholder
gdn_2_cond_1_identity?
gdn_2/cond_1/IdentityIdentitygdn_2_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity"7
gdn_2_cond_1_identitygdn_2/cond_1/Identity:output:0*(
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
2encoder_analysis_layer_2_gdn_2_cond_1_false_195959O
Kencoder_analysis_layer_2_gdn_2_cond_1_cond_encoder_analysis_layer_2_biasadd1
-encoder_analysis_layer_2_gdn_2_cond_1_equal_x2
.encoder_analysis_layer_2_gdn_2_cond_1_identity?
'encoder/analysis/layer_2/gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'encoder/analysis/layer_2/gdn_2/cond_1/x?
+encoder/analysis/layer_2/gdn_2/cond_1/EqualEqual-encoder_analysis_layer_2_gdn_2_cond_1_equal_x0encoder/analysis/layer_2/gdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_2/gdn_2/cond_1/Equal?
*encoder/analysis/layer_2/gdn_2/cond_1/condStatelessIf/encoder/analysis/layer_2/gdn_2/cond_1/Equal:z:0Kencoder_analysis_layer_2_gdn_2_cond_1_cond_encoder_analysis_layer_2_biasadd-encoder_analysis_layer_2_gdn_2_cond_1_equal_x*
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
7encoder_analysis_layer_2_gdn_2_cond_1_cond_false_195968*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_2_gdn_2_cond_1_cond_true_1959672,
*encoder/analysis/layer_2/gdn_2/cond_1/cond?
3encoder/analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity3encoder/analysis/layer_2/gdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity?
.encoder/analysis/layer_2/gdn_2/cond_1/IdentityIdentity<encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_1/Identity"i
.encoder_analysis_layer_2_gdn_2_cond_1_identity7encoder/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
 layer_2_gdn_2_cond_1_true_1991281
-layer_2_gdn_2_cond_1_identity_layer_2_biasadd$
 layer_2_gdn_2_cond_1_placeholder!
layer_2_gdn_2_cond_1_identity?
layer_2/gdn_2/cond_1/IdentityIdentity-layer_2_gdn_2_cond_1_identity_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity"G
layer_2_gdn_2_cond_1_identity&layer_2/gdn_2/cond_1/Identity:output:0*(
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
%layer_2_gdn_2_cond_2_cond_true_1987968
4layer_2_gdn_2_cond_2_cond_sqrt_layer_2_gdn_2_biasadd)
%layer_2_gdn_2_cond_2_cond_placeholder&
"layer_2_gdn_2_cond_2_cond_identity?
layer_2/gdn_2/cond_2/cond/SqrtSqrt4layer_2_gdn_2_cond_2_cond_sqrt_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_2/gdn_2/cond_2/cond/Sqrt?
"layer_2/gdn_2/cond_2/cond/IdentityIdentity"layer_2/gdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_2/cond/Identity"Q
"layer_2_gdn_2_cond_2_cond_identity+layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
M
gdn_1_cond_true_199427
gdn_1_cond_placeholder

gdn_1_cond_identity
f
gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_1/cond/Constr
gdn_1/cond/IdentityIdentitygdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity"3
gdn_1_cond_identitygdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
!layer_1_gdn_1_cond_2_false_1986523
/layer_1_gdn_1_cond_2_cond_layer_1_gdn_1_biasadd 
layer_1_gdn_1_cond_2_equal_x!
layer_1_gdn_1_cond_2_identityu
layer_1/gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_1/gdn_1/cond_2/x?
layer_1/gdn_1/cond_2/EqualEquallayer_1_gdn_1_cond_2_equal_xlayer_1/gdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/cond_2/Equal?
layer_1/gdn_1/cond_2/condStatelessIflayer_1/gdn_1/cond_2/Equal:z:0/layer_1_gdn_1_cond_2_cond_layer_1_gdn_1_biasaddlayer_1_gdn_1_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_1_gdn_1_cond_2_cond_false_198661*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_2_cond_true_1986602
layer_1/gdn_1/cond_2/cond?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity"layer_1/gdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity?
layer_1/gdn_1/cond_2/IdentityIdentity+layer_1/gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity"G
layer_1_gdn_1_cond_2_identity&layer_1/gdn_1/cond_2/Identity:output:0*(
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
6encoder_analysis_layer_2_gdn_2_cond_1_cond_true_195967S
Oencoder_analysis_layer_2_gdn_2_cond_1_cond_abs_encoder_analysis_layer_2_biasadd:
6encoder_analysis_layer_2_gdn_2_cond_1_cond_placeholder7
3encoder_analysis_layer_2_gdn_2_cond_1_cond_identity?
.encoder/analysis/layer_2/gdn_2/cond_1/cond/AbsAbsOencoder_analysis_layer_2_gdn_2_cond_1_cond_abs_encoder_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_1/cond/Abs?
3encoder/analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity2encoder/analysis/layer_2/gdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity"s
3encoder_analysis_layer_2_gdn_2_cond_1_cond_identity<encoder/analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
W
gdn_2_cond_false_196438#
gdn_2_cond_identity_gdn_2_equal

gdn_2_cond_identity
x
gdn_2/cond/IdentityIdentitygdn_2_cond_identity_gdn_2_equal*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity"3
gdn_2_cond_identitygdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_196616

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y?
truedivRealDivinputstruediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
truedivy
IdentityIdentitytruediv:z:0*
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
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_198028H
Danalysis_layer_0_gdn_0_cond_1_cond_cond_pow_analysis_layer_0_biasadd1
-analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y4
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
+analysis/layer_0/gdn_0/cond_1/cond/cond/powPowDanalysis_layer_0_gdn_0_cond_1_cond_cond_pow_analysis_layer_0_biasadd-analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/cond/pow?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity/analysis/layer_0/gdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"m
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
;encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_true_195977[
Wencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_square_encoder_analysis_layer_2_biasadd?
;encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_placeholder<
8encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_identity?
6encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/SquareSquareWencoder_analysis_layer_2_gdn_2_cond_1_cond_cond_square_encoder_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Square?
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity:encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_identityAencoder/analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
w
layer_2_gdn_2_cond_false_1986943
/layer_2_gdn_2_cond_identity_layer_2_gdn_2_equal

layer_2_gdn_2_cond_identity
?
layer_2/gdn_2/cond/IdentityIdentity/layer_2_gdn_2_cond_identity_layer_2_gdn_2_equal*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity"C
layer_2_gdn_2_cond_identity$layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
#gdn_1_cond_1_cond_cond_false_196304&
"gdn_1_cond_1_cond_cond_pow_biasadd 
gdn_1_cond_1_cond_cond_pow_y#
gdn_1_cond_1_cond_cond_identity?
gdn_1/cond_1/cond/cond/powPow"gdn_1_cond_1_cond_cond_pow_biasaddgdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/cond/pow?
gdn_1/cond_1/cond/cond/IdentityIdentitygdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity"K
gdn_1_cond_1_cond_cond_identity(gdn_1/cond_1/cond/cond/Identity:output:0*(
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
 layer_2_gdn_2_cond_1_true_1987041
-layer_2_gdn_2_cond_1_identity_layer_2_biasadd$
 layer_2_gdn_2_cond_1_placeholder!
layer_2_gdn_2_cond_1_identity?
layer_2/gdn_2/cond_1/IdentityIdentity-layer_2_gdn_2_cond_1_identity_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity"G
layer_2_gdn_2_cond_1_identity&layer_2/gdn_2/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197401

inputs
analysis_197327"
analysis_197329:	?
analysis_197331:	?
analysis_197333#
analysis_197335:
??
analysis_197337
analysis_197339
analysis_197341:	?
analysis_197343
analysis_197345
analysis_197347
analysis_197349#
analysis_197351:
??
analysis_197353:	?
analysis_197355#
analysis_197357:
??
analysis_197359
analysis_197361
analysis_197363:	?
analysis_197365
analysis_197367
analysis_197369
analysis_197371#
analysis_197373:
??
analysis_197375:	?
analysis_197377#
analysis_197379:
??
analysis_197381
analysis_197383
analysis_197385:	?
analysis_197387
analysis_197389
analysis_197391
analysis_197393#
analysis_197395:
??
analysis_197397:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinputsanalysis_197327analysis_197329analysis_197331analysis_197333analysis_197335analysis_197337analysis_197339analysis_197341analysis_197343analysis_197345analysis_197347analysis_197349analysis_197351analysis_197353analysis_197355analysis_197357analysis_197359analysis_197361analysis_197363analysis_197365analysis_197367analysis_197369analysis_197371analysis_197373analysis_197375analysis_197377analysis_197379analysis_197381analysis_197383analysis_197385analysis_197387analysis_197389analysis_197391analysis_197393analysis_197395analysis_197397*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1969372"
 analysis/StatefulPartitionedCall?
IdentityIdentity)analysis/StatefulPartitionedCall:output:0!^analysis/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2D
 analysis/StatefulPartitionedCall analysis/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
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
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197247

inputs
analysis_197173"
analysis_197175:	?
analysis_197177:	?
analysis_197179#
analysis_197181:
??
analysis_197183
analysis_197185
analysis_197187:	?
analysis_197189
analysis_197191
analysis_197193
analysis_197195#
analysis_197197:
??
analysis_197199:	?
analysis_197201#
analysis_197203:
??
analysis_197205
analysis_197207
analysis_197209:	?
analysis_197211
analysis_197213
analysis_197215
analysis_197217#
analysis_197219:
??
analysis_197221:	?
analysis_197223#
analysis_197225:
??
analysis_197227
analysis_197229
analysis_197231:	?
analysis_197233
analysis_197235
analysis_197237
analysis_197239#
analysis_197241:
??
analysis_197243:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinputsanalysis_197173analysis_197175analysis_197177analysis_197179analysis_197181analysis_197183analysis_197185analysis_197187analysis_197189analysis_197191analysis_197193analysis_197195analysis_197197analysis_197199analysis_197201analysis_197203analysis_197205analysis_197207analysis_197209analysis_197211analysis_197213analysis_197215analysis_197217analysis_197219analysis_197221analysis_197223analysis_197225analysis_197227analysis_197229analysis_197231analysis_197233analysis_197235analysis_197237analysis_197239analysis_197241analysis_197243*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1967792"
 analysis/StatefulPartitionedCall?
IdentityIdentity)analysis/StatefulPartitionedCall:output:0!^analysis/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2D
 analysis/StatefulPartitionedCall analysis/StatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
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
?
?
*analysis_layer_2_gdn_2_cond_2_false_198364E
Aanalysis_layer_2_gdn_2_cond_2_cond_analysis_layer_2_gdn_2_biasadd)
%analysis_layer_2_gdn_2_cond_2_equal_x*
&analysis_layer_2_gdn_2_cond_2_identity?
analysis/layer_2/gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_2/gdn_2/cond_2/x?
#analysis/layer_2/gdn_2/cond_2/EqualEqual%analysis_layer_2_gdn_2_cond_2_equal_x(analysis/layer_2/gdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_2/gdn_2/cond_2/Equal?
"analysis/layer_2/gdn_2/cond_2/condStatelessIf'analysis/layer_2/gdn_2/cond_2/Equal:z:0Aanalysis_layer_2_gdn_2_cond_2_cond_analysis_layer_2_gdn_2_biasadd%analysis_layer_2_gdn_2_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_2_gdn_2_cond_2_cond_false_198373*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_2_cond_true_1983722$
"analysis/layer_2/gdn_2/cond_2/cond?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity"Y
&analysis_layer_2_gdn_2_cond_2_identity/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
.analysis_layer_0_gdn_0_cond_1_cond_true_197593C
?analysis_layer_0_gdn_0_cond_1_cond_abs_analysis_layer_0_biasadd2
.analysis_layer_0_gdn_0_cond_1_cond_placeholder/
+analysis_layer_0_gdn_0_cond_1_cond_identity?
&analysis/layer_0/gdn_0/cond_1/cond/AbsAbs?analysis_layer_0_gdn_0_cond_1_cond_abs_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/cond/Abs?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity"c
+analysis_layer_0_gdn_0_cond_1_cond_identity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
gdn_1_cond_2_cond_false_199531'
#gdn_1_cond_2_cond_pow_gdn_1_biasadd
gdn_1_cond_2_cond_pow_y
gdn_1_cond_2_cond_identity?
gdn_1/cond_2/cond/powPow#gdn_1_cond_2_cond_pow_gdn_1_biasaddgdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/pow?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity"A
gdn_1_cond_2_cond_identity#gdn_1/cond_2/cond/Identity:output:0*(
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
%layer_2_gdn_2_cond_1_cond_true_1991371
-layer_2_gdn_2_cond_1_cond_abs_layer_2_biasadd)
%layer_2_gdn_2_cond_1_cond_placeholder&
"layer_2_gdn_2_cond_1_cond_identity?
layer_2/gdn_2/cond_1/cond/AbsAbs-layer_2_gdn_2_cond_1_cond_abs_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/cond/Abs?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity!layer_2/gdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity"Q
"layer_2_gdn_2_cond_1_cond_identity+layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
;encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_true_195705[
Wencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_square_encoder_analysis_layer_0_biasadd?
;encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_placeholder<
8encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
6encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/SquareSquareWencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_square_encoder_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Square?
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity:encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_identityAencoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
#gdn_2_cond_1_cond_cond_false_199602&
"gdn_2_cond_1_cond_cond_pow_biasadd 
gdn_2_cond_1_cond_cond_pow_y#
gdn_2_cond_1_cond_cond_identity?
gdn_2/cond_1/cond/cond/powPow"gdn_2_cond_1_cond_cond_pow_biasaddgdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/cond/pow?
gdn_2/cond_1/cond/cond/IdentityIdentitygdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity"K
gdn_2_cond_1_cond_cond_identity(gdn_2/cond_1/cond/cond/Identity:output:0*(
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
/encoder_analysis_layer_1_gdn_1_cond_true_1958113
/encoder_analysis_layer_1_gdn_1_cond_placeholder
0
,encoder_analysis_layer_1_gdn_1_cond_identity
?
)encoder/analysis/layer_1/gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2+
)encoder/analysis/layer_1/gdn_1/cond/Const?
,encoder/analysis/layer_1/gdn_1/cond/IdentityIdentity2encoder/analysis/layer_1/gdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_1/gdn_1/cond/Identity"e
,encoder_analysis_layer_1_gdn_1_cond_identity5encoder/analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
)analysis_layer_2_gdn_2_cond_1_true_198280C
?analysis_layer_2_gdn_2_cond_1_identity_analysis_layer_2_biasadd-
)analysis_layer_2_gdn_2_cond_1_placeholder*
&analysis_layer_2_gdn_2_cond_1_identity?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity?analysis_layer_2_gdn_2_cond_1_identity_analysis_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity"Y
&analysis_layer_2_gdn_2_cond_1_identity/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
%layer_0_gdn_0_cond_2_cond_true_1985248
4layer_0_gdn_0_cond_2_cond_sqrt_layer_0_gdn_0_biasadd)
%layer_0_gdn_0_cond_2_cond_placeholder&
"layer_0_gdn_0_cond_2_cond_identity?
layer_0/gdn_0/cond_2/cond/SqrtSqrt4layer_0_gdn_0_cond_2_cond_sqrt_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/gdn_0/cond_2/cond/Sqrt?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity"layer_0/gdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity"Q
"layer_0_gdn_0_cond_2_cond_identity+layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_197739K
Ganalysis_layer_1_gdn_1_cond_1_cond_cond_square_analysis_layer_1_biasadd7
3analysis_layer_1_gdn_1_cond_1_cond_cond_placeholder4
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
.analysis/layer_1/gdn_1/cond_1/cond/cond/SquareSquareGanalysis_layer_1_gdn_1_cond_1_cond_cond_square_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_1/gdn_1/cond_1/cond/cond/Square?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity2analysis/layer_1/gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"m
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
)analysis_layer_2_gdn_2_cond_2_true_198363I
Eanalysis_layer_2_gdn_2_cond_2_identity_analysis_layer_2_gdn_2_biasadd-
)analysis_layer_2_gdn_2_cond_2_placeholder*
&analysis_layer_2_gdn_2_cond_2_identity?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentityEanalysis_layer_2_gdn_2_cond_2_identity_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity"Y
&analysis_layer_2_gdn_2_cond_2_identity/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
M
gdn_0_cond_true_199283
gdn_0_cond_placeholder

gdn_0_cond_identity
f
gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_0/cond/Constr
gdn_0/cond/IdentityIdentitygdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity"3
gdn_0_cond_identitygdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
 layer_1_gdn_1_cond_2_true_1990757
3layer_1_gdn_1_cond_2_identity_layer_1_gdn_1_biasadd$
 layer_1_gdn_1_cond_2_placeholder!
layer_1_gdn_1_cond_2_identity?
layer_1/gdn_1/cond_2/IdentityIdentity3layer_1_gdn_1_cond_2_identity_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity"G
layer_1_gdn_1_cond_2_identity&layer_1/gdn_1/cond_2/Identity:output:0*(
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
gdn_1_cond_1_cond_true_199447!
gdn_1_cond_1_cond_abs_biasadd!
gdn_1_cond_1_cond_placeholder
gdn_1_cond_1_cond_identity?
gdn_1/cond_1/cond/AbsAbsgdn_1_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Abs?
gdn_1/cond_1/cond/IdentityIdentitygdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity"A
gdn_1_cond_1_cond_identity#gdn_1/cond_1/cond/Identity:output:0*(
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
M
gdn_1_cond_true_196273
gdn_1_cond_placeholder

gdn_1_cond_identity
f
gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_1/cond/Constr
gdn_1/cond/IdentityIdentitygdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity"3
gdn_1_cond_identitygdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
#gdn_0_cond_1_cond_cond_false_196140&
"gdn_0_cond_1_cond_cond_pow_biasadd 
gdn_0_cond_1_cond_cond_pow_y#
gdn_0_cond_1_cond_cond_identity?
gdn_0/cond_1/cond/cond/powPow"gdn_0_cond_1_cond_cond_pow_biasaddgdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/cond/pow?
gdn_0/cond_1/cond/cond/IdentityIdentitygdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity"K
gdn_0_cond_1_cond_cond_identity(gdn_0/cond_1/cond/cond/Identity:output:0*(
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
&layer_0_gdn_0_cond_1_cond_false_1984422
.layer_0_gdn_0_cond_1_cond_cond_layer_0_biasadd%
!layer_0_gdn_0_cond_1_cond_equal_x&
"layer_0_gdn_0_cond_1_cond_identity
layer_0/gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_0/gdn_0/cond_1/cond/x?
layer_0/gdn_0/cond_1/cond/EqualEqual!layer_0_gdn_0_cond_1_cond_equal_x$layer_0/gdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_0/gdn_0/cond_1/cond/Equal?
layer_0/gdn_0/cond_1/cond/condStatelessIf#layer_0/gdn_0/cond_1/cond/Equal:z:0.layer_0_gdn_0_cond_1_cond_cond_layer_0_biasadd!layer_0_gdn_0_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_0_gdn_0_cond_1_cond_cond_false_198452*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_0_gdn_0_cond_1_cond_cond_true_1984512 
layer_0/gdn_0/cond_1/cond/cond?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity'layer_0/gdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity"Q
"layer_0_gdn_0_cond_1_cond_identity+layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
.analysis_layer_2_gdn_2_cond_2_cond_true_197948J
Fanalysis_layer_2_gdn_2_cond_2_cond_sqrt_analysis_layer_2_gdn_2_biasadd2
.analysis_layer_2_gdn_2_cond_2_cond_placeholder/
+analysis_layer_2_gdn_2_cond_2_cond_identity?
'analysis/layer_2/gdn_2/cond_2/cond/SqrtSqrtFanalysis_layer_2_gdn_2_cond_2_cond_sqrt_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2)
'analysis/layer_2/gdn_2/cond_2/cond/Sqrt?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity"c
+analysis_layer_2_gdn_2_cond_2_cond_identity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
0encoder_analysis_layer_2_gdn_2_cond_false_195948U
Qencoder_analysis_layer_2_gdn_2_cond_identity_encoder_analysis_layer_2_gdn_2_equal
0
,encoder_analysis_layer_2_gdn_2_cond_identity
?
,encoder/analysis/layer_2/gdn_2/cond/IdentityIdentityQencoder_analysis_layer_2_gdn_2_cond_identity_encoder_analysis_layer_2_gdn_2_equal*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_2/gdn_2/cond/Identity"e
,encoder_analysis_layer_2_gdn_2_cond_identity5encoder/analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
gdn_2_cond_1_cond_true_199591!
gdn_2_cond_1_cond_abs_biasadd!
gdn_2_cond_1_cond_placeholder
gdn_2_cond_1_cond_identity?
gdn_2/cond_1/cond/AbsAbsgdn_2_cond_1_cond_abs_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Abs?
gdn_2/cond_1/cond/IdentityIdentitygdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity"A
gdn_2_cond_1_cond_identity#gdn_2/cond_1/cond/Identity:output:0*(
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
gdn_1_cond_2_cond_false_196377'
#gdn_1_cond_2_cond_pow_gdn_1_biasadd
gdn_1_cond_2_cond_pow_y
gdn_1_cond_2_cond_identity?
gdn_1/cond_2/cond/powPow#gdn_1_cond_2_cond_pow_gdn_1_biasaddgdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/pow?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity"A
gdn_1_cond_2_cond_identity#gdn_1/cond_2/cond/Identity:output:0*(
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
(__inference_encoder_layer_call_fn_197476
input_1
unknown
	unknown_0:	?
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

unknown_33:
??

unknown_34:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_encoder_layer_call_and_return_conditional_losses_1974012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
!layer_2_gdn_2_cond_1_false_199129-
)layer_2_gdn_2_cond_1_cond_layer_2_biasadd 
layer_2_gdn_2_cond_1_equal_x!
layer_2_gdn_2_cond_1_identityu
layer_2/gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/cond_1/x?
layer_2/gdn_2/cond_1/EqualEquallayer_2_gdn_2_cond_1_equal_xlayer_2/gdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/cond_1/Equal?
layer_2/gdn_2/cond_1/condStatelessIflayer_2/gdn_2/cond_1/Equal:z:0)layer_2_gdn_2_cond_1_cond_layer_2_biasaddlayer_2_gdn_2_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_2_gdn_2_cond_1_cond_false_199138*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_1_cond_true_1991372
layer_2/gdn_2/cond_1/cond?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity"layer_2/gdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity?
layer_2/gdn_2/cond_1/IdentityIdentity+layer_2/gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity"G
layer_2_gdn_2_cond_1_identity&layer_2/gdn_2/cond_1/Identity:output:0*(
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
6encoder_analysis_layer_1_gdn_1_cond_1_cond_true_195831S
Oencoder_analysis_layer_1_gdn_1_cond_1_cond_abs_encoder_analysis_layer_1_biasadd:
6encoder_analysis_layer_1_gdn_1_cond_1_cond_placeholder7
3encoder_analysis_layer_1_gdn_1_cond_1_cond_identity?
.encoder/analysis/layer_1/gdn_1/cond_1/cond/AbsAbsOencoder_analysis_layer_1_gdn_1_cond_1_cond_abs_encoder_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_1/cond/Abs?
3encoder/analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity2encoder/analysis/layer_1/gdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity"s
3encoder_analysis_layer_1_gdn_1_cond_1_cond_identity<encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
!layer_0_gdn_0_cond_1_false_198433-
)layer_0_gdn_0_cond_1_cond_layer_0_biasadd 
layer_0_gdn_0_cond_1_equal_x!
layer_0_gdn_0_cond_1_identityu
layer_0/gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/cond_1/x?
layer_0/gdn_0/cond_1/EqualEquallayer_0_gdn_0_cond_1_equal_xlayer_0/gdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/cond_1/Equal?
layer_0/gdn_0/cond_1/condStatelessIflayer_0/gdn_0/cond_1/Equal:z:0)layer_0_gdn_0_cond_1_cond_layer_0_biasaddlayer_0_gdn_0_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_0_gdn_0_cond_1_cond_false_198442*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_1_cond_true_1984412
layer_0/gdn_0/cond_1/cond?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity"layer_0/gdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity?
layer_0/gdn_0/cond_1/IdentityIdentity+layer_0/gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity"G
layer_0_gdn_0_cond_1_identity&layer_0/gdn_0/cond_1/Identity:output:0*(
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
;encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_true_195841[
Wencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_square_encoder_analysis_layer_1_biasadd?
;encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_placeholder<
8encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
6encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/SquareSquareWencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_square_encoder_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????28
6encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Square?
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity:encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_identityAencoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
&layer_2_gdn_2_cond_1_cond_false_1987142
.layer_2_gdn_2_cond_1_cond_cond_layer_2_biasadd%
!layer_2_gdn_2_cond_1_cond_equal_x&
"layer_2_gdn_2_cond_1_cond_identity
layer_2/gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_2/gdn_2/cond_1/cond/x?
layer_2/gdn_2/cond_1/cond/EqualEqual!layer_2_gdn_2_cond_1_cond_equal_x$layer_2/gdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_2/gdn_2/cond_1/cond/Equal?
layer_2/gdn_2/cond_1/cond/condStatelessIf#layer_2/gdn_2/cond_1/cond/Equal:z:0.layer_2_gdn_2_cond_1_cond_cond_layer_2_biasadd!layer_2_gdn_2_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_2_gdn_2_cond_1_cond_cond_false_198724*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_2_gdn_2_cond_1_cond_cond_true_1987232 
layer_2/gdn_2/cond_1/cond/cond?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity'layer_2/gdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity"Q
"layer_2_gdn_2_cond_1_cond_identity+layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
%layer_0_gdn_0_cond_1_cond_true_1988651
-layer_0_gdn_0_cond_1_cond_abs_layer_0_biasadd)
%layer_0_gdn_0_cond_1_cond_placeholder&
"layer_0_gdn_0_cond_1_cond_identity?
layer_0/gdn_0/cond_1/cond/AbsAbs-layer_0_gdn_0_cond_1_cond_abs_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/cond/Abs?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity!layer_0/gdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity"Q
"layer_0_gdn_0_cond_1_cond_identity+layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
$__inference_signature_wrapper_197555
input_1
unknown
	unknown_0:	?
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

unknown_33:
??

unknown_34:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_1960812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
e
layer_0_gdn_0_cond_true_198845"
layer_0_gdn_0_cond_placeholder

layer_0_gdn_0_cond_identity
v
layer_0/gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_0/gdn_0/cond/Const?
layer_0/gdn_0/cond/IdentityIdentity!layer_0/gdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity"C
layer_0_gdn_0_cond_identity$layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
'analysis_layer_1_gdn_1_cond_true_197709+
'analysis_layer_1_gdn_1_cond_placeholder
(
$analysis_layer_1_gdn_1_cond_identity
?
!analysis/layer_1/gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_1/gdn_1/cond/Const?
$analysis/layer_1/gdn_1/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity"U
$analysis_layer_1_gdn_1_cond_identity-analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_199263

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y?
truedivRealDivinputstruediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
truedivy
IdentityIdentitytruediv:z:0*
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
gdn_1_cond_2_cond_true_196376(
$gdn_1_cond_2_cond_sqrt_gdn_1_biasadd!
gdn_1_cond_2_cond_placeholder
gdn_1_cond_2_cond_identity?
gdn_1/cond_2/cond/SqrtSqrt$gdn_1_cond_2_cond_sqrt_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Sqrt?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity"A
gdn_1_cond_2_cond_identity#gdn_1/cond_2/cond/Identity:output:0*(
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
)analysis_layer_1_gdn_1_cond_1_true_197720C
?analysis_layer_1_gdn_1_cond_1_identity_analysis_layer_1_biasadd-
)analysis_layer_1_gdn_1_cond_1_placeholder*
&analysis_layer_1_gdn_1_cond_1_identity?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity?analysis_layer_1_gdn_1_cond_1_identity_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity"Y
&analysis_layer_1_gdn_1_cond_1_identity/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
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
u
gdn_1_cond_2_false_196368#
gdn_1_cond_2_cond_gdn_1_biasadd
gdn_1_cond_2_equal_x
gdn_1_cond_2_identitye
gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_1/cond_2/x?
gdn_1/cond_2/EqualEqualgdn_1_cond_2_equal_xgdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_2/Equal?
gdn_1/cond_2/condStatelessIfgdn_1/cond_2/Equal:z:0gdn_1_cond_2_cond_gdn_1_biasaddgdn_1_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_1_cond_2_cond_false_196377*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_2_cond_true_1963762
gdn_1/cond_2/cond?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity?
gdn_1/cond_2/IdentityIdentity#gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity"7
gdn_1_cond_2_identitygdn_1/cond_2/Identity:output:0*(
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
)analysis_layer_0_gdn_0_cond_2_true_198091I
Eanalysis_layer_0_gdn_0_cond_2_identity_analysis_layer_0_gdn_0_biasadd-
)analysis_layer_0_gdn_0_cond_2_placeholder*
&analysis_layer_0_gdn_0_cond_2_identity?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentityEanalysis_layer_0_gdn_0_cond_2_identity_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity"Y
&analysis_layer_0_gdn_0_cond_2_identity/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
gdn_0_cond_2_cond_false_199387'
#gdn_0_cond_2_cond_pow_gdn_0_biasadd
gdn_0_cond_2_cond_pow_y
gdn_0_cond_2_cond_identity?
gdn_0/cond_2/cond/powPow#gdn_0_cond_2_cond_pow_gdn_0_biasaddgdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/pow?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity"A
gdn_0_cond_2_cond_identity#gdn_0/cond_2/cond/Identity:output:0*(
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
0encoder_analysis_layer_0_gdn_0_cond_false_195676U
Qencoder_analysis_layer_0_gdn_0_cond_identity_encoder_analysis_layer_0_gdn_0_equal
0
,encoder_analysis_layer_0_gdn_0_cond_identity
?
,encoder/analysis/layer_0/gdn_0/cond/IdentityIdentityQencoder_analysis_layer_0_gdn_0_cond_identity_encoder_analysis_layer_0_gdn_0_equal*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_0/gdn_0/cond/Identity"e
,encoder_analysis_layer_0_gdn_0_cond_identity5encoder/analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
"gdn_1_cond_1_cond_cond_true_196303)
%gdn_1_cond_1_cond_cond_square_biasadd&
"gdn_1_cond_1_cond_cond_placeholder#
gdn_1_cond_1_cond_cond_identity?
gdn_1/cond_1/cond/cond/SquareSquare%gdn_1_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/cond/Square?
gdn_1/cond_1/cond/cond/IdentityIdentity!gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity"K
gdn_1_cond_1_cond_cond_identity(gdn_1/cond_1/cond/cond/Identity:output:0*(
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
e
layer_2_gdn_2_cond_true_198693"
layer_2_gdn_2_cond_placeholder

layer_2_gdn_2_cond_identity
v
layer_2/gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_2/gdn_2/cond/Const?
layer_2/gdn_2/cond/IdentityIdentity!layer_2/gdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity"C
layer_2_gdn_2_cond_identity$layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
 layer_2_gdn_2_cond_2_true_1987877
3layer_2_gdn_2_cond_2_identity_layer_2_gdn_2_biasadd$
 layer_2_gdn_2_cond_2_placeholder!
layer_2_gdn_2_cond_2_identity?
layer_2/gdn_2/cond_2/IdentityIdentity3layer_2_gdn_2_cond_2_identity_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity"G
layer_2_gdn_2_cond_2_identity&layer_2/gdn_2/cond_2/Identity:output:0*(
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
(analysis_layer_0_gdn_0_cond_false_197574E
Aanalysis_layer_0_gdn_0_cond_identity_analysis_layer_0_gdn_0_equal
(
$analysis_layer_0_gdn_0_cond_identity
?
$analysis/layer_0/gdn_0/cond/IdentityIdentityAanalysis_layer_0_gdn_0_cond_identity_analysis_layer_0_gdn_0_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity"U
$analysis_layer_0_gdn_0_cond_identity-analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
??
?
D__inference_analysis_layer_call_and_return_conditional_losses_198827

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	?6
'layer_0_biasadd_readvariableop_resource:	?
layer_0_gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y
layer_0_gdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??6
'layer_1_biasadd_readvariableop_resource:	?
layer_1_gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y
layer_1_gdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??6
'layer_2_biasadd_readvariableop_resource:	?
layer_2_gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y
layer_2_gdn_2_equal_1_x
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
??6
'layer_3_biasadd_readvariableop_resource:	?
identity??layer_0/BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?layer_1/BiasAdd/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOpi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
lambda/truediv/y?
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda/truediv?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
layer_0/Conv2DConv2Dlambda/truediv:z:0layer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_0/Conv2D?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAddlayer_0/Conv2D:output:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/BiasAddg
layer_0/gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/x?
layer_0/gdn_0/EqualEquallayer_0_gdn_0_equal_xlayer_0/gdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/Equal?
layer_0/gdn_0/condStatelessIflayer_0/gdn_0/Equal:z:0layer_0/gdn_0/Equal:z:0*
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
 *2
else_branch#R!
layer_0_gdn_0_cond_false_198422*
output_shapes
: *1
then_branch"R 
layer_0_gdn_0_cond_true_1984212
layer_0/gdn_0/cond?
layer_0/gdn_0/cond/IdentityIdentitylayer_0/gdn_0/cond:output:0*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity?
layer_0/gdn_0/cond_1StatelessIf$layer_0/gdn_0/cond/Identity:output:0layer_0/BiasAdd:output:0layer_0_gdn_0_equal_x*
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
 *4
else_branch%R#
!layer_0_gdn_0_cond_1_false_198433*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_1_true_1984322
layer_0/gdn_0/cond_1?
layer_0/gdn_0/cond_1/IdentityIdentitylayer_0/gdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198478*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198488*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
layer_0/gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/gdn_0/Reshape/shape?
layer_0/gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0$layer_0/gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/gdn_0/Reshape?
layer_0/gdn_0/convolutionConv2D&layer_0/gdn_0/cond_1/Identity:output:0layer_0/gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_0/gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198502*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
layer_0/gdn_0/BiasAddBiasAdd"layer_0/gdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/BiasAddk
layer_0/gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/x_1?
layer_0/gdn_0/Equal_1Equallayer_0_gdn_0_equal_1_xlayer_0/gdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/Equal_1?
layer_0/gdn_0/cond_2StatelessIflayer_0/gdn_0/Equal_1:z:0layer_0/gdn_0/BiasAdd:output:0layer_0_gdn_0_equal_1_x*
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
 *4
else_branch%R#
!layer_0_gdn_0_cond_2_false_198516*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_2_true_1985152
layer_0/gdn_0/cond_2?
layer_0/gdn_0/cond_2/IdentityIdentitylayer_0/gdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity?
layer_0/gdn_0/truedivRealDivlayer_0/BiasAdd:output:0&layer_0/gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/truediv?
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
layer_1/kernel/Reshape?
layer_1/Conv2DConv2Dlayer_0/gdn_0/truediv:z:0layer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_1/Conv2D?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/Conv2D:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/BiasAddg
layer_1/gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/x?
layer_1/gdn_1/EqualEquallayer_1_gdn_1_equal_xlayer_1/gdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/Equal?
layer_1/gdn_1/condStatelessIflayer_1/gdn_1/Equal:z:0layer_1/gdn_1/Equal:z:0*
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
 *2
else_branch#R!
layer_1_gdn_1_cond_false_198558*
output_shapes
: *1
then_branch"R 
layer_1_gdn_1_cond_true_1985572
layer_1/gdn_1/cond?
layer_1/gdn_1/cond/IdentityIdentitylayer_1/gdn_1/cond:output:0*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity?
layer_1/gdn_1/cond_1StatelessIf$layer_1/gdn_1/cond/Identity:output:0layer_1/BiasAdd:output:0layer_1_gdn_1_equal_x*
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
 *4
else_branch%R#
!layer_1_gdn_1_cond_1_false_198569*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_1_true_1985682
layer_1/gdn_1/cond_1?
layer_1/gdn_1/cond_1/IdentityIdentitylayer_1/gdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198614*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198624*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
layer_1/gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/gdn_1/Reshape/shape?
layer_1/gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0$layer_1/gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/gdn_1/Reshape?
layer_1/gdn_1/convolutionConv2D&layer_1/gdn_1/cond_1/Identity:output:0layer_1/gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_1/gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198638*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
layer_1/gdn_1/BiasAddBiasAdd"layer_1/gdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/BiasAddk
layer_1/gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/x_1?
layer_1/gdn_1/Equal_1Equallayer_1_gdn_1_equal_1_xlayer_1/gdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/Equal_1?
layer_1/gdn_1/cond_2StatelessIflayer_1/gdn_1/Equal_1:z:0layer_1/gdn_1/BiasAdd:output:0layer_1_gdn_1_equal_1_x*
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
 *4
else_branch%R#
!layer_1_gdn_1_cond_2_false_198652*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_2_true_1986512
layer_1/gdn_1/cond_2?
layer_1/gdn_1/cond_2/IdentityIdentitylayer_1/gdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity?
layer_1/gdn_1/truedivRealDivlayer_1/BiasAdd:output:0&layer_1/gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/truediv?
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
layer_2/kernel/Reshape?
layer_2/Conv2DConv2Dlayer_1/gdn_1/truediv:z:0layer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_2/Conv2D?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAddlayer_2/Conv2D:output:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/BiasAddg
layer_2/gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/x?
layer_2/gdn_2/EqualEquallayer_2_gdn_2_equal_xlayer_2/gdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/Equal?
layer_2/gdn_2/condStatelessIflayer_2/gdn_2/Equal:z:0layer_2/gdn_2/Equal:z:0*
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
 *2
else_branch#R!
layer_2_gdn_2_cond_false_198694*
output_shapes
: *1
then_branch"R 
layer_2_gdn_2_cond_true_1986932
layer_2/gdn_2/cond?
layer_2/gdn_2/cond/IdentityIdentitylayer_2/gdn_2/cond:output:0*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity?
layer_2/gdn_2/cond_1StatelessIf$layer_2/gdn_2/cond/Identity:output:0layer_2/BiasAdd:output:0layer_2_gdn_2_equal_x*
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
 *4
else_branch%R#
!layer_2_gdn_2_cond_1_false_198705*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_1_true_1987042
layer_2/gdn_2/cond_1?
layer_2/gdn_2/cond_1/IdentityIdentitylayer_2/gdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198750*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198760*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
layer_2/gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/gdn_2/Reshape/shape?
layer_2/gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0$layer_2/gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/gdn_2/Reshape?
layer_2/gdn_2/convolutionConv2D&layer_2/gdn_2/cond_1/Identity:output:0layer_2/gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_2/gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198774*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
layer_2/gdn_2/BiasAddBiasAdd"layer_2/gdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/BiasAddk
layer_2/gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/x_1?
layer_2/gdn_2/Equal_1Equallayer_2_gdn_2_equal_1_xlayer_2/gdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/Equal_1?
layer_2/gdn_2/cond_2StatelessIflayer_2/gdn_2/Equal_1:z:0layer_2/gdn_2/BiasAdd:output:0layer_2_gdn_2_equal_1_x*
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
 *4
else_branch%R#
!layer_2_gdn_2_cond_2_false_198788*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_2_true_1987872
layer_2/gdn_2/cond_2?
layer_2/gdn_2/cond_2/IdentityIdentitylayer_2/gdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity?
layer_2/gdn_2/truedivRealDivlayer_2/BiasAdd:output:0&layer_2/gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/truediv?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
layer_3/Conv2DConv2Dlayer_2/gdn_2/truediv:z:0layer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_3/Conv2D?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAddlayer_3/Conv2D:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_3/BiasAdd?
IdentityIdentitylayer_3/BiasAdd:output:0^layer_0/BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2@
layer_0/BiasAdd/ReadVariableOplayer_0/BiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
gdn_0_cond_2_cond_false_196213'
#gdn_0_cond_2_cond_pow_gdn_0_biasadd
gdn_0_cond_2_cond_pow_y
gdn_0_cond_2_cond_identity?
gdn_0/cond_2/cond/powPow#gdn_0_cond_2_cond_pow_gdn_0_biasaddgdn_0_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/pow?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity"A
gdn_0_cond_2_cond_identity#gdn_0/cond_2/cond/Identity:output:0*(
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
gdn_2_cond_1_cond_false_196458"
gdn_2_cond_1_cond_cond_biasadd
gdn_2_cond_1_cond_equal_x
gdn_2_cond_1_cond_identityo
gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_2/cond_1/cond/x?
gdn_2/cond_1/cond/EqualEqualgdn_2_cond_1_cond_equal_xgdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_1/cond/Equal?
gdn_2/cond_1/cond/condStatelessIfgdn_2/cond_1/cond/Equal:z:0gdn_2_cond_1_cond_cond_biasaddgdn_2_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_2_cond_1_cond_cond_false_196468*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_2_cond_1_cond_cond_true_1964672
gdn_2/cond_1/cond/cond?
gdn_2/cond_1/cond/cond/IdentityIdentitygdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity?
gdn_2/cond_1/cond/IdentityIdentity(gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity"A
gdn_2_cond_1_cond_identity#gdn_2/cond_1/cond/Identity:output:0*(
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
2encoder_analysis_layer_1_gdn_1_cond_2_false_195906U
Qencoder_analysis_layer_1_gdn_1_cond_2_cond_encoder_analysis_layer_1_gdn_1_biasadd1
-encoder_analysis_layer_1_gdn_1_cond_2_equal_x2
.encoder_analysis_layer_1_gdn_1_cond_2_identity?
'encoder/analysis/layer_1/gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder/analysis/layer_1/gdn_1/cond_2/x?
+encoder/analysis/layer_1/gdn_1/cond_2/EqualEqual-encoder_analysis_layer_1_gdn_1_cond_2_equal_x0encoder/analysis/layer_1/gdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_1/gdn_1/cond_2/Equal?
*encoder/analysis/layer_1/gdn_1/cond_2/condStatelessIf/encoder/analysis/layer_1/gdn_1/cond_2/Equal:z:0Qencoder_analysis_layer_1_gdn_1_cond_2_cond_encoder_analysis_layer_1_gdn_1_biasadd-encoder_analysis_layer_1_gdn_1_cond_2_equal_x*
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
7encoder_analysis_layer_1_gdn_1_cond_2_cond_false_195915*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_1_gdn_1_cond_2_cond_true_1959142,
*encoder/analysis/layer_1/gdn_1/cond_2/cond?
3encoder/analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity3encoder/analysis/layer_1/gdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity?
.encoder/analysis/layer_1/gdn_1/cond_2/IdentityIdentity<encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_2/Identity"i
.encoder_analysis_layer_1_gdn_1_cond_2_identity7encoder/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
gdn_1_cond_1_cond_false_199448"
gdn_1_cond_1_cond_cond_biasadd
gdn_1_cond_1_cond_equal_x
gdn_1_cond_1_cond_identityo
gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_1/cond_1/cond/x?
gdn_1/cond_1/cond/EqualEqualgdn_1_cond_1_cond_equal_xgdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_1/cond/Equal?
gdn_1/cond_1/cond/condStatelessIfgdn_1/cond_1/cond/Equal:z:0gdn_1_cond_1_cond_cond_biasaddgdn_1_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_1_cond_1_cond_cond_false_199458*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_1_cond_1_cond_cond_true_1994572
gdn_1/cond_1/cond/cond?
gdn_1/cond_1/cond/cond/IdentityIdentitygdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity?
gdn_1/cond_1/cond/IdentityIdentity(gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/Identity"A
gdn_1_cond_1_cond_identity#gdn_1/cond_1/cond/Identity:output:0*(
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
*layer_1_gdn_1_cond_1_cond_cond_true_1990119
5layer_1_gdn_1_cond_1_cond_cond_square_layer_1_biasadd.
*layer_1_gdn_1_cond_1_cond_cond_placeholder+
'layer_1_gdn_1_cond_1_cond_cond_identity?
%layer_1/gdn_1/cond_1/cond/cond/SquareSquare5layer_1_gdn_1_cond_1_cond_cond_square_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_1/gdn_1/cond_1/cond/cond/Square?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity)layer_1/gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity"[
'layer_1_gdn_1_cond_1_cond_cond_identity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
gdn_2_cond_2_cond_true_199674(
$gdn_2_cond_2_cond_sqrt_gdn_2_biasadd!
gdn_2_cond_2_cond_placeholder
gdn_2_cond_2_cond_identity?
gdn_2/cond_2/cond/SqrtSqrt$gdn_2_cond_2_cond_sqrt_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Sqrt?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity"A
gdn_2_cond_2_cond_identity#gdn_2/cond_2/cond/Identity:output:0*(
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
M
gdn_0_cond_true_196109
gdn_0_cond_placeholder

gdn_0_cond_identity
f
gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_0/cond/Constr
gdn_0/cond/IdentityIdentitygdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity"3
gdn_0_cond_identitygdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
e
layer_1_gdn_1_cond_true_198981"
layer_1_gdn_1_cond_placeholder

layer_1_gdn_1_cond_identity
v
layer_1/gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_1/gdn_1/cond/Const?
layer_1/gdn_1/cond/IdentityIdentity!layer_1/gdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity"C
layer_1_gdn_1_cond_identity$layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
v
gdn_2_cond_1_true_196448!
gdn_2_cond_1_identity_biasadd
gdn_2_cond_1_placeholder
gdn_2_cond_1_identity?
gdn_2/cond_1/IdentityIdentitygdn_2_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity"7
gdn_2_cond_1_identitygdn_2/cond_1/Identity:output:0*(
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
*layer_2_gdn_2_cond_1_cond_cond_true_1987239
5layer_2_gdn_2_cond_1_cond_cond_square_layer_2_biasadd.
*layer_2_gdn_2_cond_1_cond_cond_placeholder+
'layer_2_gdn_2_cond_1_cond_cond_identity?
%layer_2/gdn_2/cond_1/cond/cond/SquareSquare5layer_2_gdn_2_cond_1_cond_cond_square_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_2/gdn_2/cond_1/cond/cond/Square?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity)layer_2/gdn_2/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity"[
'layer_2_gdn_2_cond_1_cond_cond_identity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
%layer_1_gdn_1_cond_2_cond_true_1990848
4layer_1_gdn_1_cond_2_cond_sqrt_layer_1_gdn_1_biasadd)
%layer_1_gdn_1_cond_2_cond_placeholder&
"layer_1_gdn_1_cond_2_cond_identity?
layer_1/gdn_1/cond_2/cond/SqrtSqrt4layer_1_gdn_1_cond_2_cond_sqrt_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_1/gdn_1/cond_2/cond/Sqrt?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity"layer_1/gdn_1/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity"Q
"layer_1_gdn_1_cond_2_cond_identity+layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
&layer_1_gdn_1_cond_1_cond_false_1985782
.layer_1_gdn_1_cond_1_cond_cond_layer_1_biasadd%
!layer_1_gdn_1_cond_1_cond_equal_x&
"layer_1_gdn_1_cond_1_cond_identity
layer_1/gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_1/gdn_1/cond_1/cond/x?
layer_1/gdn_1/cond_1/cond/EqualEqual!layer_1_gdn_1_cond_1_cond_equal_x$layer_1/gdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_1/gdn_1/cond_1/cond/Equal?
layer_1/gdn_1/cond_1/cond/condStatelessIf#layer_1/gdn_1/cond_1/cond/Equal:z:0.layer_1_gdn_1_cond_1_cond_cond_layer_1_biasadd!layer_1_gdn_1_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_1_gdn_1_cond_1_cond_cond_false_198588*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_1_gdn_1_cond_1_cond_cond_true_1985872 
layer_1/gdn_1/cond_1/cond/cond?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity'layer_1/gdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity"Q
"layer_1_gdn_1_cond_1_cond_identity+layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
gdn_0_cond_1_cond_false_199304"
gdn_0_cond_1_cond_cond_biasadd
gdn_0_cond_1_cond_equal_x
gdn_0_cond_1_cond_identityo
gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_0/cond_1/cond/x?
gdn_0/cond_1/cond/EqualEqualgdn_0_cond_1_cond_equal_xgdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_1/cond/Equal?
gdn_0/cond_1/cond/condStatelessIfgdn_0/cond_1/cond/Equal:z:0gdn_0_cond_1_cond_cond_biasaddgdn_0_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_0_cond_1_cond_cond_false_199314*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_0_cond_1_cond_cond_true_1993132
gdn_0/cond_1/cond/cond?
gdn_0/cond_1/cond/cond/IdentityIdentitygdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity?
gdn_0/cond_1/cond/IdentityIdentity(gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity"A
gdn_0_cond_1_cond_identity#gdn_0/cond_1/cond/Identity:output:0*(
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
*layer_0_gdn_0_cond_1_cond_cond_true_1984519
5layer_0_gdn_0_cond_1_cond_cond_square_layer_0_biasadd.
*layer_0_gdn_0_cond_1_cond_cond_placeholder+
'layer_0_gdn_0_cond_1_cond_cond_identity?
%layer_0/gdn_0/cond_1/cond/cond/SquareSquare5layer_0_gdn_0_cond_1_cond_cond_square_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_0/gdn_0/cond_1/cond/cond/Square?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity)layer_0/gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity"[
'layer_0_gdn_0_cond_1_cond_cond_identity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
 layer_0_gdn_0_cond_1_true_1988561
-layer_0_gdn_0_cond_1_identity_layer_0_biasadd$
 layer_0_gdn_0_cond_1_placeholder!
layer_0_gdn_0_cond_1_identity?
layer_0/gdn_0/cond_1/IdentityIdentity-layer_0_gdn_0_cond_1_identity_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity"G
layer_0_gdn_0_cond_1_identity&layer_0/gdn_0/cond_1/Identity:output:0*(
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
*analysis_layer_0_gdn_0_cond_1_false_197585?
;analysis_layer_0_gdn_0_cond_1_cond_analysis_layer_0_biasadd)
%analysis_layer_0_gdn_0_cond_1_equal_x*
&analysis_layer_0_gdn_0_cond_1_identity?
analysis/layer_0/gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_0/gdn_0/cond_1/x?
#analysis/layer_0/gdn_0/cond_1/EqualEqual%analysis_layer_0_gdn_0_cond_1_equal_x(analysis/layer_0/gdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_0/gdn_0/cond_1/Equal?
"analysis/layer_0/gdn_0/cond_1/condStatelessIf'analysis/layer_0/gdn_0/cond_1/Equal:z:0;analysis_layer_0_gdn_0_cond_1_cond_analysis_layer_0_biasadd%analysis_layer_0_gdn_0_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_0_gdn_0_cond_1_cond_false_197594*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_1_cond_true_1975932$
"analysis/layer_0/gdn_0/cond_1/cond?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity"Y
&analysis_layer_0_gdn_0_cond_1_identity/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
!layer_1_gdn_1_cond_2_false_1990763
/layer_1_gdn_1_cond_2_cond_layer_1_gdn_1_biasadd 
layer_1_gdn_1_cond_2_equal_x!
layer_1_gdn_1_cond_2_identityu
layer_1/gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
layer_1/gdn_1/cond_2/x?
layer_1/gdn_1/cond_2/EqualEquallayer_1_gdn_1_cond_2_equal_xlayer_1/gdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/cond_2/Equal?
layer_1/gdn_1/cond_2/condStatelessIflayer_1/gdn_1/cond_2/Equal:z:0/layer_1_gdn_1_cond_2_cond_layer_1_gdn_1_biasaddlayer_1_gdn_1_cond_2_equal_x*
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
 *9
else_branch*R(
&layer_1_gdn_1_cond_2_cond_false_199085*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_2_cond_true_1990842
layer_1/gdn_1/cond_2/cond?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity"layer_1/gdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity?
layer_1/gdn_1/cond_2/IdentityIdentity+layer_1/gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity"G
layer_1_gdn_1_cond_2_identity&layer_1/gdn_1/cond_2/Identity:output:0*(
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
#gdn_1_cond_1_cond_cond_false_199458&
"gdn_1_cond_1_cond_cond_pow_biasadd 
gdn_1_cond_1_cond_cond_pow_y#
gdn_1_cond_1_cond_cond_identity?
gdn_1/cond_1/cond/cond/powPow"gdn_1_cond_1_cond_cond_pow_biasaddgdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/cond/pow?
gdn_1/cond_1/cond/cond/IdentityIdentitygdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity"K
gdn_1_cond_1_cond_cond_identity(gdn_1/cond_1/cond/cond/Identity:output:0*(
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
o
gdn_2_cond_1_false_199583
gdn_2_cond_1_cond_biasadd
gdn_2_cond_1_equal_x
gdn_2_cond_1_identitye
gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_2/cond_1/x?
gdn_2/cond_1/EqualEqualgdn_2_cond_1_equal_xgdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_1/Equal?
gdn_2/cond_1/condStatelessIfgdn_2/cond_1/Equal:z:0gdn_2_cond_1_cond_biasaddgdn_2_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_2_cond_1_cond_false_199592*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_1_cond_true_1995912
gdn_2/cond_1/cond?
gdn_2/cond_1/cond/IdentityIdentitygdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity?
gdn_2/cond_1/IdentityIdentity#gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity"7
gdn_2_cond_1_identitygdn_2/cond_1/Identity:output:0*(
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
v
gdn_0_cond_1_true_199294!
gdn_0_cond_1_identity_biasadd
gdn_0_cond_1_placeholder
gdn_0_cond_1_identity?
gdn_0/cond_1/IdentityIdentitygdn_0_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity"7
gdn_0_cond_1_identitygdn_0/cond_1/Identity:output:0*(
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
/analysis_layer_1_gdn_1_cond_2_cond_false_198237I
Eanalysis_layer_1_gdn_1_cond_2_cond_pow_analysis_layer_1_gdn_1_biasadd,
(analysis_layer_1_gdn_1_cond_2_cond_pow_y/
+analysis_layer_1_gdn_1_cond_2_cond_identity?
&analysis/layer_1/gdn_1/cond_2/cond/powPowEanalysis_layer_1_gdn_1_cond_2_cond_pow_analysis_layer_1_gdn_1_biasadd(analysis_layer_1_gdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/cond/pow?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity"c
+analysis_layer_1_gdn_1_cond_2_cond_identity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197090
input_1
analysis_197016"
analysis_197018:	?
analysis_197020:	?
analysis_197022#
analysis_197024:
??
analysis_197026
analysis_197028
analysis_197030:	?
analysis_197032
analysis_197034
analysis_197036
analysis_197038#
analysis_197040:
??
analysis_197042:	?
analysis_197044#
analysis_197046:
??
analysis_197048
analysis_197050
analysis_197052:	?
analysis_197054
analysis_197056
analysis_197058
analysis_197060#
analysis_197062:
??
analysis_197064:	?
analysis_197066#
analysis_197068:
??
analysis_197070
analysis_197072
analysis_197074:	?
analysis_197076
analysis_197078
analysis_197080
analysis_197082#
analysis_197084:
??
analysis_197086:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinput_1analysis_197016analysis_197018analysis_197020analysis_197022analysis_197024analysis_197026analysis_197028analysis_197030analysis_197032analysis_197034analysis_197036analysis_197038analysis_197040analysis_197042analysis_197044analysis_197046analysis_197048analysis_197050analysis_197052analysis_197054analysis_197056analysis_197058analysis_197060analysis_197062analysis_197064analysis_197066analysis_197068analysis_197070analysis_197072analysis_197074analysis_197076analysis_197078analysis_197080analysis_197082analysis_197084analysis_197086*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1967792"
 analysis/StatefulPartitionedCall?
IdentityIdentity)analysis/StatefulPartitionedCall:output:0!^analysis/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2D
 analysis/StatefulPartitionedCall analysis/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
+layer_2_gdn_2_cond_1_cond_cond_false_1991486
2layer_2_gdn_2_cond_1_cond_cond_pow_layer_2_biasadd(
$layer_2_gdn_2_cond_1_cond_cond_pow_y+
'layer_2_gdn_2_cond_1_cond_cond_identity?
"layer_2/gdn_2/cond_1/cond/cond/powPow2layer_2_gdn_2_cond_1_cond_cond_pow_layer_2_biasadd$layer_2_gdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/cond/pow?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity&layer_2/gdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity"[
'layer_2_gdn_2_cond_1_cond_cond_identity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
/analysis_layer_1_gdn_1_cond_1_cond_false_198154D
@analysis_layer_1_gdn_1_cond_1_cond_cond_analysis_layer_1_biasadd.
*analysis_layer_1_gdn_1_cond_1_cond_equal_x/
+analysis_layer_1_gdn_1_cond_1_cond_identity?
$analysis/layer_1/gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_1/gdn_1/cond_1/cond/x?
(analysis/layer_1/gdn_1/cond_1/cond/EqualEqual*analysis_layer_1_gdn_1_cond_1_cond_equal_x-analysis/layer_1/gdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_1/gdn_1/cond_1/cond/Equal?
'analysis/layer_1/gdn_1/cond_1/cond/condStatelessIf,analysis/layer_1/gdn_1/cond_1/cond/Equal:z:0@analysis_layer_1_gdn_1_cond_1_cond_cond_analysis_layer_1_biasadd*analysis_layer_1_gdn_1_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_198164*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_1981632)
'analysis/layer_1/gdn_1/cond_1/cond/cond?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity0analysis/layer_1/gdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity"c
+analysis_layer_1_gdn_1_cond_1_cond_identity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
/analysis_layer_2_gdn_2_cond_2_cond_false_198373I
Eanalysis_layer_2_gdn_2_cond_2_cond_pow_analysis_layer_2_gdn_2_biasadd,
(analysis_layer_2_gdn_2_cond_2_cond_pow_y/
+analysis_layer_2_gdn_2_cond_2_cond_identity?
&analysis/layer_2/gdn_2/cond_2/cond/powPowEanalysis_layer_2_gdn_2_cond_2_cond_pow_analysis_layer_2_gdn_2_biasadd(analysis_layer_2_gdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/cond/pow?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity"c
+analysis_layer_2_gdn_2_cond_2_cond_identity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
w
layer_1_gdn_1_cond_false_1989823
/layer_1_gdn_1_cond_identity_layer_1_gdn_1_equal

layer_1_gdn_1_cond_identity
?
layer_1/gdn_1/cond/IdentityIdentity/layer_1_gdn_1_cond_identity_layer_1_gdn_1_equal*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity"C
layer_1_gdn_1_cond_identity$layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
u
gdn_1_cond_2_false_199522#
gdn_1_cond_2_cond_gdn_1_biasadd
gdn_1_cond_2_equal_x
gdn_1_cond_2_identitye
gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_1/cond_2/x?
gdn_1/cond_2/EqualEqualgdn_1_cond_2_equal_xgdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/cond_2/Equal?
gdn_1/cond_2/condStatelessIfgdn_1/cond_2/Equal:z:0gdn_1_cond_2_cond_gdn_1_biasaddgdn_1_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_1_cond_2_cond_false_199531*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_2_cond_true_1995302
gdn_1/cond_2/cond?
gdn_1/cond_2/cond/IdentityIdentitygdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/cond/Identity?
gdn_1/cond_2/IdentityIdentity#gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity"7
gdn_1_cond_2_identitygdn_1/cond_2/Identity:output:0*(
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
|
gdn_1_cond_2_true_199521'
#gdn_1_cond_2_identity_gdn_1_biasadd
gdn_1_cond_2_placeholder
gdn_1_cond_2_identity?
gdn_1/cond_2/IdentityIdentity#gdn_1_cond_2_identity_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity"7
gdn_1_cond_2_identitygdn_1/cond_2/Identity:output:0*(
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
+layer_2_gdn_2_cond_1_cond_cond_false_1987246
2layer_2_gdn_2_cond_1_cond_cond_pow_layer_2_biasadd(
$layer_2_gdn_2_cond_1_cond_cond_pow_y+
'layer_2_gdn_2_cond_1_cond_cond_identity?
"layer_2/gdn_2/cond_1/cond/cond/powPow2layer_2_gdn_2_cond_1_cond_cond_pow_layer_2_biasadd$layer_2_gdn_2_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/cond/pow?
'layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity&layer_2/gdn_2/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_2/gdn_2/cond_1/cond/cond/Identity"[
'layer_2_gdn_2_cond_1_cond_cond_identity0layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*(
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
&layer_1_gdn_1_cond_1_cond_false_1990022
.layer_1_gdn_1_cond_1_cond_cond_layer_1_biasadd%
!layer_1_gdn_1_cond_1_cond_equal_x&
"layer_1_gdn_1_cond_1_cond_identity
layer_1/gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_1/gdn_1/cond_1/cond/x?
layer_1/gdn_1/cond_1/cond/EqualEqual!layer_1_gdn_1_cond_1_cond_equal_x$layer_1/gdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_1/gdn_1/cond_1/cond/Equal?
layer_1/gdn_1/cond_1/cond/condStatelessIf#layer_1/gdn_1/cond_1/cond/Equal:z:0.layer_1_gdn_1_cond_1_cond_cond_layer_1_biasadd!layer_1_gdn_1_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_1_gdn_1_cond_1_cond_cond_false_199012*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_1_gdn_1_cond_1_cond_cond_true_1990112 
layer_1/gdn_1/cond_1/cond/cond?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity'layer_1/gdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity"Q
"layer_1_gdn_1_cond_1_cond_identity+layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
%layer_1_gdn_1_cond_1_cond_true_1985771
-layer_1_gdn_1_cond_1_cond_abs_layer_1_biasadd)
%layer_1_gdn_1_cond_1_cond_placeholder&
"layer_1_gdn_1_cond_1_cond_identity?
layer_1/gdn_1/cond_1/cond/AbsAbs-layer_1_gdn_1_cond_1_cond_abs_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/cond/Abs?
"layer_1/gdn_1/cond_1/cond/IdentityIdentity!layer_1/gdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/Identity"Q
"layer_1_gdn_1_cond_1_cond_identity+layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
6encoder_analysis_layer_0_gdn_0_cond_2_cond_true_195778Z
Vencoder_analysis_layer_0_gdn_0_cond_2_cond_sqrt_encoder_analysis_layer_0_gdn_0_biasadd:
6encoder_analysis_layer_0_gdn_0_cond_2_cond_placeholder7
3encoder_analysis_layer_0_gdn_0_cond_2_cond_identity?
/encoder/analysis/layer_0/gdn_0/cond_2/cond/SqrtSqrtVencoder_analysis_layer_0_gdn_0_cond_2_cond_sqrt_encoder_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????21
/encoder/analysis/layer_0/gdn_0/cond_2/cond/Sqrt?
3encoder/analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity3encoder/analysis/layer_0/gdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity"s
3encoder_analysis_layer_0_gdn_0_cond_2_cond_identity<encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*(
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
"__inference__traced_restore_199882
file_prefix,
assignvariableop_layer_0_bias:	?<
-assignvariableop_1_layer_0_gdn_0_reparam_beta:	?B
.assignvariableop_2_layer_0_gdn_0_reparam_gamma:
??9
&assignvariableop_3_layer_0_kernel_rdft:	?.
assignvariableop_4_layer_1_bias:	?<
-assignvariableop_5_layer_1_gdn_1_reparam_beta:	?B
.assignvariableop_6_layer_1_gdn_1_reparam_gamma:
??:
&assignvariableop_7_layer_1_kernel_rdft:
??.
assignvariableop_8_layer_2_bias:	?<
-assignvariableop_9_layer_2_gdn_2_reparam_beta:	?C
/assignvariableop_10_layer_2_gdn_2_reparam_gamma:
??;
'assignvariableop_11_layer_2_kernel_rdft:
??/
 assignvariableop_12_layer_3_bias:	?;
'assignvariableop_13_layer_3_kernel_rdft:
??
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
AssignVariableOp_1AssignVariableOp-assignvariableop_1_layer_0_gdn_0_reparam_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_layer_0_gdn_0_reparam_gammaIdentity_2:output:0"/device:CPU:0*
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
AssignVariableOp_5AssignVariableOp-assignvariableop_5_layer_1_gdn_1_reparam_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp.assignvariableop_6_layer_1_gdn_1_reparam_gammaIdentity_6:output:0"/device:CPU:0*
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
AssignVariableOp_9AssignVariableOp-assignvariableop_9_layer_2_gdn_2_reparam_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_layer_2_gdn_2_reparam_gammaIdentity_10:output:0"/device:CPU:0*
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
'analysis_layer_1_gdn_1_cond_true_198133+
'analysis_layer_1_gdn_1_cond_placeholder
(
$analysis_layer_1_gdn_1_cond_identity
?
!analysis/layer_1/gdn_1/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_1/gdn_1/cond/Const?
$analysis/layer_1/gdn_1/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity"U
$analysis_layer_1_gdn_1_cond_identity-analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?	
?
7encoder_analysis_layer_2_gdn_2_cond_2_cond_false_196051Y
Uencoder_analysis_layer_2_gdn_2_cond_2_cond_pow_encoder_analysis_layer_2_gdn_2_biasadd4
0encoder_analysis_layer_2_gdn_2_cond_2_cond_pow_y7
3encoder_analysis_layer_2_gdn_2_cond_2_cond_identity?
.encoder/analysis/layer_2/gdn_2/cond_2/cond/powPowUencoder_analysis_layer_2_gdn_2_cond_2_cond_pow_encoder_analysis_layer_2_gdn_2_biasadd0encoder_analysis_layer_2_gdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_2/cond/pow?
3encoder/analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity2encoder/analysis/layer_2/gdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity"s
3encoder_analysis_layer_2_gdn_2_cond_2_cond_identity<encoder/analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*(
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
gdn_2_cond_2_cond_false_196541'
#gdn_2_cond_2_cond_pow_gdn_2_biasadd
gdn_2_cond_2_cond_pow_y
gdn_2_cond_2_cond_identity?
gdn_2/cond_2/cond/powPow#gdn_2_cond_2_cond_pow_gdn_2_biasaddgdn_2_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/pow?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity"A
gdn_2_cond_2_cond_identity#gdn_2/cond_2/cond/Identity:output:0*(
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
)analysis_layer_1_gdn_1_cond_2_true_198227I
Eanalysis_layer_1_gdn_1_cond_2_identity_analysis_layer_1_gdn_1_biasadd-
)analysis_layer_1_gdn_1_cond_2_placeholder*
&analysis_layer_1_gdn_1_cond_2_identity?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentityEanalysis_layer_1_gdn_1_cond_2_identity_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity"Y
&analysis_layer_1_gdn_1_cond_2_identity/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
%layer_0_gdn_0_cond_1_cond_true_1984411
-layer_0_gdn_0_cond_1_cond_abs_layer_0_biasadd)
%layer_0_gdn_0_cond_1_cond_placeholder&
"layer_0_gdn_0_cond_1_cond_identity?
layer_0/gdn_0/cond_1/cond/AbsAbs-layer_0_gdn_0_cond_1_cond_abs_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/cond/Abs?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity!layer_0/gdn_0/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity"Q
"layer_0_gdn_0_cond_1_cond_identity+layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
M
gdn_2_cond_true_199571
gdn_2_cond_placeholder

gdn_2_cond_identity
f
gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
gdn_2/cond/Constr
gdn_2/cond/IdentityIdentitygdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity"3
gdn_2_cond_identitygdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
)analysis_layer_0_gdn_0_cond_2_true_197667I
Eanalysis_layer_0_gdn_0_cond_2_identity_analysis_layer_0_gdn_0_biasadd-
)analysis_layer_0_gdn_0_cond_2_placeholder*
&analysis_layer_0_gdn_0_cond_2_identity?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentityEanalysis_layer_0_gdn_0_cond_2_identity_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity"Y
&analysis_layer_0_gdn_0_cond_2_identity/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
o
gdn_0_cond_1_false_199295
gdn_0_cond_1_cond_biasadd
gdn_0_cond_1_equal_x
gdn_0_cond_1_identitye
gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_0/cond_1/x?
gdn_0/cond_1/EqualEqualgdn_0_cond_1_equal_xgdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_1/Equal?
gdn_0/cond_1/condStatelessIfgdn_0/cond_1/Equal:z:0gdn_0_cond_1_cond_biasaddgdn_0_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_0_cond_1_cond_false_199304*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_1_cond_true_1993032
gdn_0/cond_1/cond?
gdn_0/cond_1/cond/IdentityIdentitygdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity?
gdn_0/cond_1/IdentityIdentity#gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity"7
gdn_0_cond_1_identitygdn_0/cond_1/Identity:output:0*(
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
/analysis_layer_2_gdn_2_cond_1_cond_false_197866D
@analysis_layer_2_gdn_2_cond_1_cond_cond_analysis_layer_2_biasadd.
*analysis_layer_2_gdn_2_cond_1_cond_equal_x/
+analysis_layer_2_gdn_2_cond_1_cond_identity?
$analysis/layer_2/gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_2/gdn_2/cond_1/cond/x?
(analysis/layer_2/gdn_2/cond_1/cond/EqualEqual*analysis_layer_2_gdn_2_cond_1_cond_equal_x-analysis/layer_2/gdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_2/gdn_2/cond_1/cond/Equal?
'analysis/layer_2/gdn_2/cond_1/cond/condStatelessIf,analysis/layer_2/gdn_2/cond_1/cond/Equal:z:0@analysis_layer_2_gdn_2_cond_1_cond_cond_analysis_layer_2_biasadd*analysis_layer_2_gdn_2_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_197876*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_1978752)
'analysis/layer_2/gdn_2/cond_1/cond/cond?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity0analysis/layer_2/gdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity"c
+analysis_layer_2_gdn_2_cond_1_cond_identity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_197740H
Danalysis_layer_1_gdn_1_cond_1_cond_cond_pow_analysis_layer_1_biasadd1
-analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y4
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
+analysis/layer_1/gdn_1/cond_1/cond/cond/powPowDanalysis_layer_1_gdn_1_cond_1_cond_cond_pow_analysis_layer_1_biasadd-analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/cond/pow?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity/analysis/layer_1/gdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"m
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
o
gdn_0_cond_1_false_196121
gdn_0_cond_1_cond_biasadd
gdn_0_cond_1_equal_x
gdn_0_cond_1_identitye
gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_0/cond_1/x?
gdn_0/cond_1/EqualEqualgdn_0_cond_1_equal_xgdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/cond_1/Equal?
gdn_0/cond_1/condStatelessIfgdn_0/cond_1/Equal:z:0gdn_0_cond_1_cond_biasaddgdn_0_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_0_cond_1_cond_false_196130*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_1_cond_true_1961292
gdn_0/cond_1/cond?
gdn_0/cond_1/cond/IdentityIdentitygdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/Identity?
gdn_0/cond_1/IdentityIdentity#gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity"7
gdn_0_cond_1_identitygdn_0/cond_1/Identity:output:0*(
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
2encoder_analysis_layer_0_gdn_0_cond_2_false_195770U
Qencoder_analysis_layer_0_gdn_0_cond_2_cond_encoder_analysis_layer_0_gdn_0_biasadd1
-encoder_analysis_layer_0_gdn_0_cond_2_equal_x2
.encoder_analysis_layer_0_gdn_0_cond_2_identity?
'encoder/analysis/layer_0/gdn_0/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'encoder/analysis/layer_0/gdn_0/cond_2/x?
+encoder/analysis/layer_0/gdn_0/cond_2/EqualEqual-encoder_analysis_layer_0_gdn_0_cond_2_equal_x0encoder/analysis/layer_0/gdn_0/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_0/gdn_0/cond_2/Equal?
*encoder/analysis/layer_0/gdn_0/cond_2/condStatelessIf/encoder/analysis/layer_0/gdn_0/cond_2/Equal:z:0Qencoder_analysis_layer_0_gdn_0_cond_2_cond_encoder_analysis_layer_0_gdn_0_biasadd-encoder_analysis_layer_0_gdn_0_cond_2_equal_x*
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
7encoder_analysis_layer_0_gdn_0_cond_2_cond_false_195779*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_0_gdn_0_cond_2_cond_true_1957782,
*encoder/analysis/layer_0/gdn_0/cond_2/cond?
3encoder/analysis/layer_0/gdn_0/cond_2/cond/IdentityIdentity3encoder/analysis/layer_0/gdn_0/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity?
.encoder/analysis/layer_0/gdn_0/cond_2/IdentityIdentity<encoder/analysis/layer_0/gdn_0/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_2/Identity"i
.encoder_analysis_layer_0_gdn_0_cond_2_identity7encoder/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
'analysis_layer_2_gdn_2_cond_true_198269+
'analysis_layer_2_gdn_2_cond_placeholder
(
$analysis_layer_2_gdn_2_cond_identity
?
!analysis/layer_2/gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_2/gdn_2/cond/Const?
$analysis/layer_2/gdn_2/cond/IdentityIdentity*analysis/layer_2/gdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity"U
$analysis_layer_2_gdn_2_cond_identity-analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
e
layer_2_gdn_2_cond_true_199117"
layer_2_gdn_2_cond_placeholder

layer_2_gdn_2_cond_identity
v
layer_2/gdn_2/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
layer_2/gdn_2/cond/Const?
layer_2/gdn_2/cond/IdentityIdentity!layer_2/gdn_2/cond/Const:output:0*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity"C
layer_2_gdn_2_cond_identity$layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_199257

inputs
identity[
	truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
	truediv/y?
truedivRealDivinputstruediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
truedivy
IdentityIdentitytruediv:z:0*
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
v
gdn_0_cond_1_true_196120!
gdn_0_cond_1_identity_biasadd
gdn_0_cond_1_placeholder
gdn_0_cond_1_identity?
gdn_0/cond_1/IdentityIdentitygdn_0_cond_1_identity_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity"7
gdn_0_cond_1_identitygdn_0/cond_1/Identity:output:0*(
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
*analysis_layer_2_gdn_2_cond_2_false_197940E
Aanalysis_layer_2_gdn_2_cond_2_cond_analysis_layer_2_gdn_2_biasadd)
%analysis_layer_2_gdn_2_cond_2_equal_x*
&analysis_layer_2_gdn_2_cond_2_identity?
analysis/layer_2/gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_2/gdn_2/cond_2/x?
#analysis/layer_2/gdn_2/cond_2/EqualEqual%analysis_layer_2_gdn_2_cond_2_equal_x(analysis/layer_2/gdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_2/gdn_2/cond_2/Equal?
"analysis/layer_2/gdn_2/cond_2/condStatelessIf'analysis/layer_2/gdn_2/cond_2/Equal:z:0Aanalysis_layer_2_gdn_2_cond_2_cond_analysis_layer_2_gdn_2_biasadd%analysis_layer_2_gdn_2_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_2_gdn_2_cond_2_cond_false_197949*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_2_cond_true_1979482$
"analysis/layer_2/gdn_2/cond_2/cond?
+analysis/layer_2/gdn_2/cond_2/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_2/cond/Identity?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentity4analysis/layer_2/gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity"Y
&analysis_layer_2_gdn_2_cond_2_identity/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
)analysis_layer_0_gdn_0_cond_1_true_197584C
?analysis_layer_0_gdn_0_cond_1_identity_analysis_layer_0_biasadd-
)analysis_layer_0_gdn_0_cond_1_placeholder*
&analysis_layer_0_gdn_0_cond_1_identity?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity?analysis_layer_0_gdn_0_cond_1_identity_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity"Y
&analysis_layer_0_gdn_0_cond_1_identity/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
?
C__inference_encoder_layer_call_and_return_conditional_losses_197979

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	??
0analysis_layer_0_biasadd_readvariableop_resource:	?"
analysis_layer_0_gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y$
 analysis_layer_0_gdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
???
0analysis_layer_1_biasadd_readvariableop_resource:	?"
analysis_layer_1_gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y$
 analysis_layer_1_gdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
???
0analysis_layer_2_biasadd_readvariableop_resource:	?"
analysis_layer_2_gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y$
 analysis_layer_2_gdn_2_equal_1_x
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
???
0analysis_layer_3_biasadd_readvariableop_resource:	?
identity??'analysis/layer_0/BiasAdd/ReadVariableOp?'analysis/layer_1/BiasAdd/ReadVariableOp?'analysis/layer_2/BiasAdd/ReadVariableOp?'analysis/layer_3/BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp{
analysis/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
analysis/lambda/truediv/y?
analysis/lambda/truedivRealDivinputs"analysis/lambda/truediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
analysis/lambda/truediv?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
analysis/layer_0/Conv2DConv2Danalysis/lambda/truediv:z:0layer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_0/Conv2D?
'analysis/layer_0/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_0/BiasAdd/ReadVariableOp?
analysis/layer_0/BiasAddBiasAdd analysis/layer_0/Conv2D:output:0/analysis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_0/BiasAddy
analysis/layer_0/gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_0/gdn_0/x?
analysis/layer_0/gdn_0/EqualEqualanalysis_layer_0_gdn_0_equal_x!analysis/layer_0/gdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_0/gdn_0/Equal?
analysis/layer_0/gdn_0/condStatelessIf analysis/layer_0/gdn_0/Equal:z:0 analysis/layer_0/gdn_0/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_0_gdn_0_cond_false_197574*
output_shapes
: *:
then_branch+R)
'analysis_layer_0_gdn_0_cond_true_1975732
analysis/layer_0/gdn_0/cond?
$analysis/layer_0/gdn_0/cond/IdentityIdentity$analysis/layer_0/gdn_0/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity?
analysis/layer_0/gdn_0/cond_1StatelessIf-analysis/layer_0/gdn_0/cond/Identity:output:0!analysis/layer_0/BiasAdd:output:0analysis_layer_0_gdn_0_equal_x*
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
 *=
else_branch.R,
*analysis_layer_0_gdn_0_cond_1_false_197585*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_1_true_1975842
analysis/layer_0/gdn_0/cond_1?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity&analysis/layer_0/gdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197630*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197640*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
$analysis/layer_0/gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_0/gdn_0/Reshape/shape?
analysis/layer_0/gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0-analysis/layer_0/gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_0/gdn_0/Reshape?
"analysis/layer_0/gdn_0/convolutionConv2D/analysis/layer_0/gdn_0/cond_1/Identity:output:0'analysis/layer_0/gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_0/gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197654*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
analysis/layer_0/gdn_0/BiasAddBiasAdd+analysis/layer_0/gdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_0/gdn_0/BiasAdd}
analysis/layer_0/gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_0/gdn_0/x_1?
analysis/layer_0/gdn_0/Equal_1Equal analysis_layer_0_gdn_0_equal_1_x#analysis/layer_0/gdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_0/gdn_0/Equal_1?
analysis/layer_0/gdn_0/cond_2StatelessIf"analysis/layer_0/gdn_0/Equal_1:z:0'analysis/layer_0/gdn_0/BiasAdd:output:0 analysis_layer_0_gdn_0_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_0_gdn_0_cond_2_false_197668*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_2_true_1976672
analysis/layer_0/gdn_0/cond_2?
&analysis/layer_0/gdn_0/cond_2/IdentityIdentity&analysis/layer_0/gdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_2/Identity?
analysis/layer_0/gdn_0/truedivRealDiv!analysis/layer_0/BiasAdd:output:0/analysis/layer_0/gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_0/gdn_0/truediv?
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
layer_1/kernel/Reshape?
analysis/layer_1/Conv2DConv2D"analysis/layer_0/gdn_0/truediv:z:0layer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_1/Conv2D?
'analysis/layer_1/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_1/BiasAdd/ReadVariableOp?
analysis/layer_1/BiasAddBiasAdd analysis/layer_1/Conv2D:output:0/analysis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_1/BiasAddy
analysis/layer_1/gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_1/gdn_1/x?
analysis/layer_1/gdn_1/EqualEqualanalysis_layer_1_gdn_1_equal_x!analysis/layer_1/gdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_1/gdn_1/Equal?
analysis/layer_1/gdn_1/condStatelessIf analysis/layer_1/gdn_1/Equal:z:0 analysis/layer_1/gdn_1/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_1_gdn_1_cond_false_197710*
output_shapes
: *:
then_branch+R)
'analysis_layer_1_gdn_1_cond_true_1977092
analysis/layer_1/gdn_1/cond?
$analysis/layer_1/gdn_1/cond/IdentityIdentity$analysis/layer_1/gdn_1/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity?
analysis/layer_1/gdn_1/cond_1StatelessIf-analysis/layer_1/gdn_1/cond/Identity:output:0!analysis/layer_1/BiasAdd:output:0analysis_layer_1_gdn_1_equal_x*
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
 *=
else_branch.R,
*analysis_layer_1_gdn_1_cond_1_false_197721*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_1_true_1977202
analysis/layer_1/gdn_1/cond_1?
&analysis/layer_1/gdn_1/cond_1/IdentityIdentity&analysis/layer_1/gdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197766*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197776*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
$analysis/layer_1/gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_1/gdn_1/Reshape/shape?
analysis/layer_1/gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0-analysis/layer_1/gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_1/gdn_1/Reshape?
"analysis/layer_1/gdn_1/convolutionConv2D/analysis/layer_1/gdn_1/cond_1/Identity:output:0'analysis/layer_1/gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_1/gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197790*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
analysis/layer_1/gdn_1/BiasAddBiasAdd+analysis/layer_1/gdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_1/gdn_1/BiasAdd}
analysis/layer_1/gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_1/gdn_1/x_1?
analysis/layer_1/gdn_1/Equal_1Equal analysis_layer_1_gdn_1_equal_1_x#analysis/layer_1/gdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_1/gdn_1/Equal_1?
analysis/layer_1/gdn_1/cond_2StatelessIf"analysis/layer_1/gdn_1/Equal_1:z:0'analysis/layer_1/gdn_1/BiasAdd:output:0 analysis_layer_1_gdn_1_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_1_gdn_1_cond_2_false_197804*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_2_true_1978032
analysis/layer_1/gdn_1/cond_2?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentity&analysis/layer_1/gdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity?
analysis/layer_1/gdn_1/truedivRealDiv!analysis/layer_1/BiasAdd:output:0/analysis/layer_1/gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_1/gdn_1/truediv?
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
layer_2/kernel/Reshape?
analysis/layer_2/Conv2DConv2D"analysis/layer_1/gdn_1/truediv:z:0layer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_2/Conv2D?
'analysis/layer_2/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_2/BiasAdd/ReadVariableOp?
analysis/layer_2/BiasAddBiasAdd analysis/layer_2/Conv2D:output:0/analysis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_2/BiasAddy
analysis/layer_2/gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_2/gdn_2/x?
analysis/layer_2/gdn_2/EqualEqualanalysis_layer_2_gdn_2_equal_x!analysis/layer_2/gdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
analysis/layer_2/gdn_2/Equal?
analysis/layer_2/gdn_2/condStatelessIf analysis/layer_2/gdn_2/Equal:z:0 analysis/layer_2/gdn_2/Equal:z:0*
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
 *;
else_branch,R*
(analysis_layer_2_gdn_2_cond_false_197846*
output_shapes
: *:
then_branch+R)
'analysis_layer_2_gdn_2_cond_true_1978452
analysis/layer_2/gdn_2/cond?
$analysis/layer_2/gdn_2/cond/IdentityIdentity$analysis/layer_2/gdn_2/cond:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity?
analysis/layer_2/gdn_2/cond_1StatelessIf-analysis/layer_2/gdn_2/cond/Identity:output:0!analysis/layer_2/BiasAdd:output:0analysis_layer_2_gdn_2_equal_x*
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
 *=
else_branch.R,
*analysis_layer_2_gdn_2_cond_1_false_197857*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_1_true_1978562
analysis/layer_2/gdn_2/cond_1?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity&analysis/layer_2/gdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197902*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197912*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
$analysis/layer_2/gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2&
$analysis/layer_2/gdn_2/Reshape/shape?
analysis/layer_2/gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0-analysis/layer_2/gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2 
analysis/layer_2/gdn_2/Reshape?
"analysis/layer_2/gdn_2/convolutionConv2D/analysis/layer_2/gdn_2/cond_1/Identity:output:0'analysis/layer_2/gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2$
"analysis/layer_2/gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-197926*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
analysis/layer_2/gdn_2/BiasAddBiasAdd+analysis/layer_2/gdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_2/gdn_2/BiasAdd}
analysis/layer_2/gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
analysis/layer_2/gdn_2/x_1?
analysis/layer_2/gdn_2/Equal_1Equal analysis_layer_2_gdn_2_equal_1_x#analysis/layer_2/gdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2 
analysis/layer_2/gdn_2/Equal_1?
analysis/layer_2/gdn_2/cond_2StatelessIf"analysis/layer_2/gdn_2/Equal_1:z:0'analysis/layer_2/gdn_2/BiasAdd:output:0 analysis_layer_2_gdn_2_equal_1_x*
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
 *=
else_branch.R,
*analysis_layer_2_gdn_2_cond_2_false_197940*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_2_true_1979392
analysis/layer_2/gdn_2/cond_2?
&analysis/layer_2/gdn_2/cond_2/IdentityIdentity&analysis/layer_2/gdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_2/Identity?
analysis/layer_2/gdn_2/truedivRealDiv!analysis/layer_2/BiasAdd:output:0/analysis/layer_2/gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2 
analysis/layer_2/gdn_2/truediv?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
analysis/layer_3/Conv2DConv2D"analysis/layer_2/gdn_2/truediv:z:0layer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
analysis/layer_3/Conv2D?
'analysis/layer_3/BiasAdd/ReadVariableOpReadVariableOp0analysis_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'analysis/layer_3/BiasAdd/ReadVariableOp?
analysis/layer_3/BiasAddBiasAdd analysis/layer_3/Conv2D:output:0/analysis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
analysis/layer_3/BiasAdd?
IdentityIdentity!analysis/layer_3/BiasAdd:output:0(^analysis/layer_0/BiasAdd/ReadVariableOp(^analysis/layer_1/BiasAdd/ReadVariableOp(^analysis/layer_2/BiasAdd/ReadVariableOp(^analysis/layer_3/BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2R
'analysis/layer_0/BiasAdd/ReadVariableOp'analysis/layer_0/BiasAdd/ReadVariableOp2R
'analysis/layer_1/BiasAdd/ReadVariableOp'analysis/layer_1/BiasAdd/ReadVariableOp2R
'analysis/layer_2/BiasAdd/ReadVariableOp'analysis/layer_2/BiasAdd/ReadVariableOp2R
'analysis/layer_3/BiasAdd/ReadVariableOp'analysis/layer_3/BiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_197603K
Ganalysis_layer_0_gdn_0_cond_1_cond_cond_square_analysis_layer_0_biasadd7
3analysis_layer_0_gdn_0_cond_1_cond_cond_placeholder4
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
.analysis/layer_0/gdn_0/cond_1/cond/cond/SquareSquareGanalysis_layer_0_gdn_0_cond_1_cond_cond_square_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_0/gdn_0/cond_1/cond/cond/Square?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity2analysis/layer_0/gdn_0/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"m
0analysis_layer_0_gdn_0_cond_1_cond_cond_identity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
!layer_2_gdn_2_cond_1_false_198705-
)layer_2_gdn_2_cond_1_cond_layer_2_biasadd 
layer_2_gdn_2_cond_1_equal_x!
layer_2_gdn_2_cond_1_identityu
layer_2/gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/cond_1/x?
layer_2/gdn_2/cond_1/EqualEquallayer_2_gdn_2_cond_1_equal_xlayer_2/gdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/cond_1/Equal?
layer_2/gdn_2/cond_1/condStatelessIflayer_2/gdn_2/cond_1/Equal:z:0)layer_2_gdn_2_cond_1_cond_layer_2_biasaddlayer_2_gdn_2_cond_1_equal_x*
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
 *9
else_branch*R(
&layer_2_gdn_2_cond_1_cond_false_198714*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_1_cond_true_1987132
layer_2/gdn_2/cond_1/cond?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity"layer_2/gdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity?
layer_2/gdn_2/cond_1/IdentityIdentity+layer_2/gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity"G
layer_2_gdn_2_cond_1_identity&layer_2/gdn_2/cond_1/Identity:output:0*(
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
&layer_1_gdn_1_cond_2_cond_false_1990857
3layer_1_gdn_1_cond_2_cond_pow_layer_1_gdn_1_biasadd#
layer_1_gdn_1_cond_2_cond_pow_y&
"layer_1_gdn_1_cond_2_cond_identity?
layer_1/gdn_1/cond_2/cond/powPow3layer_1_gdn_1_cond_2_cond_pow_layer_1_gdn_1_biasaddlayer_1_gdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/cond/pow?
"layer_1/gdn_1/cond_2/cond/IdentityIdentity!layer_1/gdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_2/cond/Identity"Q
"layer_1_gdn_1_cond_2_cond_identity+layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
D__inference_analysis_layer_call_and_return_conditional_losses_199251

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	?6
'layer_0_biasadd_readvariableop_resource:	?
layer_0_gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y
layer_0_gdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??6
'layer_1_biasadd_readvariableop_resource:	?
layer_1_gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y
layer_1_gdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??6
'layer_2_biasadd_readvariableop_resource:	?
layer_2_gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y
layer_2_gdn_2_equal_1_x
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
??6
'layer_3_biasadd_readvariableop_resource:	?
identity??layer_0/BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?layer_1/BiasAdd/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?layer_2/BiasAdd/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?layer_3/BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOpi
lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2
lambda/truediv/y?
lambda/truedivRealDivinputslambda/truediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
lambda/truediv?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
layer_0/Conv2DConv2Dlambda/truediv:z:0layer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_0/Conv2D?
layer_0/BiasAdd/ReadVariableOpReadVariableOp'layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_0/BiasAdd/ReadVariableOp?
layer_0/BiasAddBiasAddlayer_0/Conv2D:output:0&layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/BiasAddg
layer_0/gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/x?
layer_0/gdn_0/EqualEquallayer_0_gdn_0_equal_xlayer_0/gdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/Equal?
layer_0/gdn_0/condStatelessIflayer_0/gdn_0/Equal:z:0layer_0/gdn_0/Equal:z:0*
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
 *2
else_branch#R!
layer_0_gdn_0_cond_false_198846*
output_shapes
: *1
then_branch"R 
layer_0_gdn_0_cond_true_1988452
layer_0/gdn_0/cond?
layer_0/gdn_0/cond/IdentityIdentitylayer_0/gdn_0/cond:output:0*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity?
layer_0/gdn_0/cond_1StatelessIf$layer_0/gdn_0/cond/Identity:output:0layer_0/BiasAdd:output:0layer_0_gdn_0_equal_x*
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
 *4
else_branch%R#
!layer_0_gdn_0_cond_1_false_198857*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_1_true_1988562
layer_0/gdn_0/cond_1?
layer_0/gdn_0/cond_1/IdentityIdentitylayer_0/gdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198902*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198912*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
layer_0/gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_0/gdn_0/Reshape/shape?
layer_0/gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0$layer_0/gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_0/gdn_0/Reshape?
layer_0/gdn_0/convolutionConv2D&layer_0/gdn_0/cond_1/Identity:output:0layer_0/gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_0/gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-198926*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
layer_0/gdn_0/BiasAddBiasAdd"layer_0/gdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/BiasAddk
layer_0/gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_0/gdn_0/x_1?
layer_0/gdn_0/Equal_1Equallayer_0_gdn_0_equal_1_xlayer_0/gdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_0/gdn_0/Equal_1?
layer_0/gdn_0/cond_2StatelessIflayer_0/gdn_0/Equal_1:z:0layer_0/gdn_0/BiasAdd:output:0layer_0_gdn_0_equal_1_x*
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
 *4
else_branch%R#
!layer_0_gdn_0_cond_2_false_198940*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_2_true_1989392
layer_0/gdn_0/cond_2?
layer_0/gdn_0/cond_2/IdentityIdentitylayer_0/gdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_2/Identity?
layer_0/gdn_0/truedivRealDivlayer_0/BiasAdd:output:0&layer_0/gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/truediv?
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
layer_1/kernel/Reshape?
layer_1/Conv2DConv2Dlayer_0/gdn_0/truediv:z:0layer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_1/Conv2D?
layer_1/BiasAdd/ReadVariableOpReadVariableOp'layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_1/BiasAdd/ReadVariableOp?
layer_1/BiasAddBiasAddlayer_1/Conv2D:output:0&layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/BiasAddg
layer_1/gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/x?
layer_1/gdn_1/EqualEquallayer_1_gdn_1_equal_xlayer_1/gdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/Equal?
layer_1/gdn_1/condStatelessIflayer_1/gdn_1/Equal:z:0layer_1/gdn_1/Equal:z:0*
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
 *2
else_branch#R!
layer_1_gdn_1_cond_false_198982*
output_shapes
: *1
then_branch"R 
layer_1_gdn_1_cond_true_1989812
layer_1/gdn_1/cond?
layer_1/gdn_1/cond/IdentityIdentitylayer_1/gdn_1/cond:output:0*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity?
layer_1/gdn_1/cond_1StatelessIf$layer_1/gdn_1/cond/Identity:output:0layer_1/BiasAdd:output:0layer_1_gdn_1_equal_x*
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
 *4
else_branch%R#
!layer_1_gdn_1_cond_1_false_198993*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_1_true_1989922
layer_1/gdn_1/cond_1?
layer_1/gdn_1/cond_1/IdentityIdentitylayer_1/gdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199038*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199048*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
layer_1/gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_1/gdn_1/Reshape/shape?
layer_1/gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0$layer_1/gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_1/gdn_1/Reshape?
layer_1/gdn_1/convolutionConv2D&layer_1/gdn_1/cond_1/Identity:output:0layer_1/gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_1/gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199062*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
layer_1/gdn_1/BiasAddBiasAdd"layer_1/gdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/BiasAddk
layer_1/gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_1/gdn_1/x_1?
layer_1/gdn_1/Equal_1Equallayer_1_gdn_1_equal_1_xlayer_1/gdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_1/gdn_1/Equal_1?
layer_1/gdn_1/cond_2StatelessIflayer_1/gdn_1/Equal_1:z:0layer_1/gdn_1/BiasAdd:output:0layer_1_gdn_1_equal_1_x*
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
 *4
else_branch%R#
!layer_1_gdn_1_cond_2_false_199076*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_2_true_1990752
layer_1/gdn_1/cond_2?
layer_1/gdn_1/cond_2/IdentityIdentitylayer_1/gdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_2/Identity?
layer_1/gdn_1/truedivRealDivlayer_1/BiasAdd:output:0&layer_1/gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/truediv?
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
layer_2/kernel/Reshape?
layer_2/Conv2DConv2Dlayer_1/gdn_1/truediv:z:0layer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_2/Conv2D?
layer_2/BiasAdd/ReadVariableOpReadVariableOp'layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_2/BiasAdd/ReadVariableOp?
layer_2/BiasAddBiasAddlayer_2/Conv2D:output:0&layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/BiasAddg
layer_2/gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/x?
layer_2/gdn_2/EqualEquallayer_2_gdn_2_equal_xlayer_2/gdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/Equal?
layer_2/gdn_2/condStatelessIflayer_2/gdn_2/Equal:z:0layer_2/gdn_2/Equal:z:0*
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
 *2
else_branch#R!
layer_2_gdn_2_cond_false_199118*
output_shapes
: *1
then_branch"R 
layer_2_gdn_2_cond_true_1991172
layer_2/gdn_2/cond?
layer_2/gdn_2/cond/IdentityIdentitylayer_2/gdn_2/cond:output:0*
T0
*
_output_shapes
: 2
layer_2/gdn_2/cond/Identity?
layer_2/gdn_2/cond_1StatelessIf$layer_2/gdn_2/cond/Identity:output:0layer_2/BiasAdd:output:0layer_2_gdn_2_equal_x*
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
 *4
else_branch%R#
!layer_2_gdn_2_cond_1_false_199129*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_1_true_1991282
layer_2/gdn_2/cond_1?
layer_2/gdn_2/cond_1/IdentityIdentitylayer_2/gdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199174*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199184*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
layer_2/gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_2/gdn_2/Reshape/shape?
layer_2/gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0$layer_2/gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_2/gdn_2/Reshape?
layer_2/gdn_2/convolutionConv2D&layer_2/gdn_2/cond_1/Identity:output:0layer_2/gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
layer_2/gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199198*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
layer_2/gdn_2/BiasAddBiasAdd"layer_2/gdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/BiasAddk
layer_2/gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
layer_2/gdn_2/x_1?
layer_2/gdn_2/Equal_1Equallayer_2_gdn_2_equal_1_xlayer_2/gdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
layer_2/gdn_2/Equal_1?
layer_2/gdn_2/cond_2StatelessIflayer_2/gdn_2/Equal_1:z:0layer_2/gdn_2/BiasAdd:output:0layer_2_gdn_2_equal_1_x*
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
 *4
else_branch%R#
!layer_2_gdn_2_cond_2_false_199212*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_2_true_1992112
layer_2/gdn_2/cond_2?
layer_2/gdn_2/cond_2/IdentityIdentitylayer_2/gdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_2/Identity?
layer_2/gdn_2/truedivRealDivlayer_2/BiasAdd:output:0&layer_2/gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/truediv?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
layer_3/Conv2DConv2Dlayer_2/gdn_2/truediv:z:0layer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
layer_3/Conv2D?
layer_3/BiasAdd/ReadVariableOpReadVariableOp'layer_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
layer_3/BiasAdd/ReadVariableOp?
layer_3/BiasAddBiasAddlayer_3/Conv2D:output:0&layer_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_3/BiasAdd?
IdentityIdentitylayer_3/BiasAdd:output:0^layer_0/BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp^layer_1/BiasAdd/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp^layer_2/BiasAdd/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp^layer_3/BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2@
layer_0/BiasAdd/ReadVariableOplayer_0/BiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2@
layer_1/BiasAdd/ReadVariableOplayer_1/BiasAdd/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2@
layer_2/BiasAdd/ReadVariableOplayer_2/BiasAdd/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2@
layer_3/BiasAdd/ReadVariableOplayer_3/BiasAdd/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
?
 layer_1_gdn_1_cond_1_true_1989921
-layer_1_gdn_1_cond_1_identity_layer_1_biasadd$
 layer_1_gdn_1_cond_1_placeholder!
layer_1_gdn_1_cond_1_identity?
layer_1/gdn_1/cond_1/IdentityIdentity-layer_1_gdn_1_cond_1_identity_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_1/gdn_1/cond_1/Identity"G
layer_1_gdn_1_cond_1_identity&layer_1/gdn_1/cond_1/Identity:output:0*(
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
*analysis_layer_0_gdn_0_cond_1_false_198009?
;analysis_layer_0_gdn_0_cond_1_cond_analysis_layer_0_biasadd)
%analysis_layer_0_gdn_0_cond_1_equal_x*
&analysis_layer_0_gdn_0_cond_1_identity?
analysis/layer_0/gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_0/gdn_0/cond_1/x?
#analysis/layer_0/gdn_0/cond_1/EqualEqual%analysis_layer_0_gdn_0_cond_1_equal_x(analysis/layer_0/gdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_0/gdn_0/cond_1/Equal?
"analysis/layer_0/gdn_0/cond_1/condStatelessIf'analysis/layer_0/gdn_0/cond_1/Equal:z:0;analysis_layer_0_gdn_0_cond_1_cond_analysis_layer_0_biasadd%analysis_layer_0_gdn_0_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_0_gdn_0_cond_1_cond_false_198018*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_1_cond_true_1980172$
"analysis/layer_0/gdn_0/cond_1/cond?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity+analysis/layer_0/gdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity"Y
&analysis_layer_0_gdn_0_cond_1_identity/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
(analysis_layer_2_gdn_2_cond_false_198270E
Aanalysis_layer_2_gdn_2_cond_identity_analysis_layer_2_gdn_2_equal
(
$analysis_layer_2_gdn_2_cond_identity
?
$analysis/layer_2/gdn_2/cond/IdentityIdentityAanalysis_layer_2_gdn_2_cond_identity_analysis_layer_2_gdn_2_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_2/gdn_2/cond/Identity"U
$analysis_layer_2_gdn_2_cond_identity-analysis/layer_2/gdn_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
1encoder_analysis_layer_1_gdn_1_cond_2_true_195905Y
Uencoder_analysis_layer_1_gdn_1_cond_2_identity_encoder_analysis_layer_1_gdn_1_biasadd5
1encoder_analysis_layer_1_gdn_1_cond_2_placeholder2
.encoder_analysis_layer_1_gdn_1_cond_2_identity?
.encoder/analysis/layer_1/gdn_1/cond_2/IdentityIdentityUencoder_analysis_layer_1_gdn_1_cond_2_identity_encoder_analysis_layer_1_gdn_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_2/Identity"i
.encoder_analysis_layer_1_gdn_1_cond_2_identity7encoder/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
?
C__inference_layer_3_layer_call_and_return_conditional_losses_199713

inputs
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
Conv2DConv2Dinputslayer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

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
u
gdn_2_cond_2_false_196532#
gdn_2_cond_2_cond_gdn_2_biasadd
gdn_2_cond_2_equal_x
gdn_2_cond_2_identitye
gdn_2/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
gdn_2/cond_2/x?
gdn_2/cond_2/EqualEqualgdn_2_cond_2_equal_xgdn_2/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_2/Equal?
gdn_2/cond_2/condStatelessIfgdn_2/cond_2/Equal:z:0gdn_2_cond_2_cond_gdn_2_biasaddgdn_2_cond_2_equal_x*
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
 *1
else_branch"R 
gdn_2_cond_2_cond_false_196541*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_2_cond_true_1965402
gdn_2/cond_2/cond?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity?
gdn_2/cond_2/IdentityIdentity#gdn_2/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity"7
gdn_2_cond_2_identitygdn_2/cond_2/Identity:output:0*(
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
1encoder_analysis_layer_2_gdn_2_cond_2_true_196041Y
Uencoder_analysis_layer_2_gdn_2_cond_2_identity_encoder_analysis_layer_2_gdn_2_biasadd5
1encoder_analysis_layer_2_gdn_2_cond_2_placeholder2
.encoder_analysis_layer_2_gdn_2_cond_2_identity?
.encoder/analysis/layer_2/gdn_2/cond_2/IdentityIdentityUencoder_analysis_layer_2_gdn_2_cond_2_identity_encoder_analysis_layer_2_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_2/Identity"i
.encoder_analysis_layer_2_gdn_2_cond_2_identity7encoder/analysis/layer_2/gdn_2/cond_2/Identity:output:0*(
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
.analysis_layer_1_gdn_1_cond_1_cond_true_198153C
?analysis_layer_1_gdn_1_cond_1_cond_abs_analysis_layer_1_biasadd2
.analysis_layer_1_gdn_1_cond_1_cond_placeholder/
+analysis_layer_1_gdn_1_cond_1_cond_identity?
&analysis/layer_1/gdn_1/cond_1/cond/AbsAbs?analysis_layer_1_gdn_1_cond_1_cond_abs_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_1/cond/Abs?
+analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_1/cond/Identity"c
+analysis_layer_1_gdn_1_cond_1_cond_identity4analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
1encoder_analysis_layer_0_gdn_0_cond_1_true_195686S
Oencoder_analysis_layer_0_gdn_0_cond_1_identity_encoder_analysis_layer_0_biasadd5
1encoder_analysis_layer_0_gdn_0_cond_1_placeholder2
.encoder_analysis_layer_0_gdn_0_cond_1_identity?
.encoder/analysis/layer_0/gdn_0/cond_1/IdentityIdentityOencoder_analysis_layer_0_gdn_0_cond_1_identity_encoder_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_1/Identity"i
.encoder_analysis_layer_0_gdn_0_cond_1_identity7encoder/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
w
layer_0_gdn_0_cond_false_1988463
/layer_0_gdn_0_cond_identity_layer_0_gdn_0_equal

layer_0_gdn_0_cond_identity
?
layer_0/gdn_0/cond/IdentityIdentity/layer_0_gdn_0_cond_identity_layer_0_gdn_0_equal*
T0
*
_output_shapes
: 2
layer_0/gdn_0/cond/Identity"C
layer_0_gdn_0_cond_identity$layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
&layer_0_gdn_0_cond_1_cond_false_1988662
.layer_0_gdn_0_cond_1_cond_cond_layer_0_biasadd%
!layer_0_gdn_0_cond_1_cond_equal_x&
"layer_0_gdn_0_cond_1_cond_identity
layer_0/gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
layer_0/gdn_0/cond_1/cond/x?
layer_0/gdn_0/cond_1/cond/EqualEqual!layer_0_gdn_0_cond_1_cond_equal_x$layer_0/gdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2!
layer_0/gdn_0/cond_1/cond/Equal?
layer_0/gdn_0/cond_1/cond/condStatelessIf#layer_0/gdn_0/cond_1/cond/Equal:z:0.layer_0_gdn_0_cond_1_cond_cond_layer_0_biasadd!layer_0_gdn_0_cond_1_cond_equal_x*
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
 *>
else_branch/R-
+layer_0_gdn_0_cond_1_cond_cond_false_198876*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_0_gdn_0_cond_1_cond_cond_true_1988752 
layer_0/gdn_0/cond_1/cond/cond?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity'layer_0/gdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity?
"layer_0/gdn_0/cond_1/cond/IdentityIdentity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/Identity"Q
"layer_0_gdn_0_cond_1_cond_identity+layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
?
7encoder_analysis_layer_1_gdn_1_cond_1_cond_false_195832T
Pencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_encoder_analysis_layer_1_biasadd6
2encoder_analysis_layer_1_gdn_1_cond_1_cond_equal_x7
3encoder_analysis_layer_1_gdn_1_cond_1_cond_identity?
,encoder/analysis/layer_1/gdn_1/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2.
,encoder/analysis/layer_1/gdn_1/cond_1/cond/x?
0encoder/analysis/layer_1/gdn_1/cond_1/cond/EqualEqual2encoder_analysis_layer_1_gdn_1_cond_1_cond_equal_x5encoder/analysis/layer_1/gdn_1/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 22
0encoder/analysis/layer_1/gdn_1/cond_1/cond/Equal?
/encoder/analysis/layer_1/gdn_1/cond_1/cond/condStatelessIf4encoder/analysis/layer_1/gdn_1/cond_1/cond/Equal:z:0Pencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_encoder_analysis_layer_1_biasadd2encoder_analysis_layer_1_gdn_1_cond_1_cond_equal_x*
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
<encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_false_195842*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_true_19584121
/encoder/analysis/layer_1/gdn_1/cond_1/cond/cond?
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity?
3encoder/analysis/layer_1/gdn_1/cond_1/cond/IdentityIdentityAencoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity"s
3encoder_analysis_layer_1_gdn_1_cond_1_cond_identity<encoder/analysis/layer_1/gdn_1/cond_1/cond/Identity:output:0*(
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
(analysis_layer_1_gdn_1_cond_false_197710E
Aanalysis_layer_1_gdn_1_cond_identity_analysis_layer_1_gdn_1_equal
(
$analysis_layer_1_gdn_1_cond_identity
?
$analysis/layer_1/gdn_1/cond/IdentityIdentityAanalysis_layer_1_gdn_1_cond_identity_analysis_layer_1_gdn_1_equal*
T0
*
_output_shapes
: 2&
$analysis/layer_1/gdn_1/cond/Identity"U
$analysis_layer_1_gdn_1_cond_identity-analysis/layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
%layer_0_gdn_0_cond_2_cond_true_1989488
4layer_0_gdn_0_cond_2_cond_sqrt_layer_0_gdn_0_biasadd)
%layer_0_gdn_0_cond_2_cond_placeholder&
"layer_0_gdn_0_cond_2_cond_identity?
layer_0/gdn_0/cond_2/cond/SqrtSqrt4layer_0_gdn_0_cond_2_cond_sqrt_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2 
layer_0/gdn_0/cond_2/cond/Sqrt?
"layer_0/gdn_0/cond_2/cond/IdentityIdentity"layer_0/gdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_2/cond/Identity"Q
"layer_0_gdn_0_cond_2_cond_identity+layer_0/gdn_0/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197167
input_1
analysis_197093"
analysis_197095:	?
analysis_197097:	?
analysis_197099#
analysis_197101:
??
analysis_197103
analysis_197105
analysis_197107:	?
analysis_197109
analysis_197111
analysis_197113
analysis_197115#
analysis_197117:
??
analysis_197119:	?
analysis_197121#
analysis_197123:
??
analysis_197125
analysis_197127
analysis_197129:	?
analysis_197131
analysis_197133
analysis_197135
analysis_197137#
analysis_197139:
??
analysis_197141:	?
analysis_197143#
analysis_197145:
??
analysis_197147
analysis_197149
analysis_197151:	?
analysis_197153
analysis_197155
analysis_197157
analysis_197159#
analysis_197161:
??
analysis_197163:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinput_1analysis_197093analysis_197095analysis_197097analysis_197099analysis_197101analysis_197103analysis_197105analysis_197107analysis_197109analysis_197111analysis_197113analysis_197115analysis_197117analysis_197119analysis_197121analysis_197123analysis_197125analysis_197127analysis_197129analysis_197131analysis_197133analysis_197135analysis_197137analysis_197139analysis_197141analysis_197143analysis_197145analysis_197147analysis_197149analysis_197151analysis_197153analysis_197155analysis_197157analysis_197159analysis_197161analysis_197163*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*0
_read_only_resource_inputs
#$*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_analysis_layer_call_and_return_conditional_losses_1969372"
 analysis/StatefulPartitionedCall?
IdentityIdentity)analysis/StatefulPartitionedCall:output:0!^analysis/StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2D
 analysis/StatefulPartitionedCall analysis/StatefulPartitionedCall:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
 layer_0_gdn_0_cond_1_true_1984321
-layer_0_gdn_0_cond_1_identity_layer_0_biasadd$
 layer_0_gdn_0_cond_1_placeholder!
layer_0_gdn_0_cond_1_identity?
layer_0/gdn_0/cond_1/IdentityIdentity-layer_0_gdn_0_cond_1_identity_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_0/gdn_0/cond_1/Identity"G
layer_0_gdn_0_cond_1_identity&layer_0/gdn_0/cond_1/Identity:output:0*(
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
gdn_0_cond_2_cond_true_199386(
$gdn_0_cond_2_cond_sqrt_gdn_0_biasadd!
gdn_0_cond_2_cond_placeholder
gdn_0_cond_2_cond_identity?
gdn_0/cond_2/cond/SqrtSqrt$gdn_0_cond_2_cond_sqrt_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Sqrt?
gdn_0/cond_2/cond/IdentityIdentitygdn_0/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/cond/Identity"A
gdn_0_cond_2_cond_identity#gdn_0/cond_2/cond/Identity:output:0*(
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
<encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_false_195706X
Tencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_pow_encoder_analysis_layer_0_biasadd9
5encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y<
8encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_identity?
3encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/powPowTencoder_analysis_layer_0_gdn_0_cond_1_cond_cond_pow_encoder_analysis_layer_0_biasadd5encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/pow?
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity7encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_identityAencoder/analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
gdn_2_cond_1_cond_false_199592"
gdn_2_cond_1_cond_cond_biasadd
gdn_2_cond_1_cond_equal_x
gdn_2_cond_1_cond_identityo
gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
gdn_2/cond_1/cond/x?
gdn_2/cond_1/cond/EqualEqualgdn_2_cond_1_cond_equal_xgdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_1/cond/Equal?
gdn_2/cond_1/cond/condStatelessIfgdn_2/cond_1/cond/Equal:z:0gdn_2_cond_1_cond_cond_biasaddgdn_2_cond_1_cond_equal_x*
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
 *6
else_branch'R%
#gdn_2_cond_1_cond_cond_false_199602*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_2_cond_1_cond_cond_true_1996012
gdn_2/cond_1/cond/cond?
gdn_2/cond_1/cond/cond/IdentityIdentitygdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_2/cond_1/cond/cond/Identity?
gdn_2/cond_1/cond/IdentityIdentity(gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity"A
gdn_2_cond_1_cond_identity#gdn_2/cond_1/cond/Identity:output:0*(
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
/analysis_layer_1_gdn_1_cond_2_cond_false_197813I
Eanalysis_layer_1_gdn_1_cond_2_cond_pow_analysis_layer_1_gdn_1_biasadd,
(analysis_layer_1_gdn_1_cond_2_cond_pow_y/
+analysis_layer_1_gdn_1_cond_2_cond_identity?
&analysis/layer_1/gdn_1/cond_2/cond/powPowEanalysis_layer_1_gdn_1_cond_2_cond_pow_analysis_layer_1_gdn_1_biasadd(analysis_layer_1_gdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/cond/pow?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity*analysis/layer_1/gdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity"c
+analysis_layer_1_gdn_1_cond_2_cond_identity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
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
*analysis_layer_2_gdn_2_cond_1_false_197857?
;analysis_layer_2_gdn_2_cond_1_cond_analysis_layer_2_biasadd)
%analysis_layer_2_gdn_2_cond_1_equal_x*
&analysis_layer_2_gdn_2_cond_1_identity?
analysis/layer_2/gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2!
analysis/layer_2/gdn_2/cond_1/x?
#analysis/layer_2/gdn_2/cond_1/EqualEqual%analysis_layer_2_gdn_2_cond_1_equal_x(analysis/layer_2/gdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_2/gdn_2/cond_1/Equal?
"analysis/layer_2/gdn_2/cond_1/condStatelessIf'analysis/layer_2/gdn_2/cond_1/Equal:z:0;analysis_layer_2_gdn_2_cond_1_cond_analysis_layer_2_biasadd%analysis_layer_2_gdn_2_cond_1_equal_x*
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
 *B
else_branch3R1
/analysis_layer_2_gdn_2_cond_1_cond_false_197866*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_1_cond_true_1978652$
"analysis/layer_2/gdn_2/cond_1/cond?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity+analysis/layer_2/gdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity?
&analysis/layer_2/gdn_2/cond_1/IdentityIdentity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_2/gdn_2/cond_1/Identity"Y
&analysis_layer_2_gdn_2_cond_1_identity/analysis/layer_2/gdn_2/cond_1/Identity:output:0*(
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
'analysis_layer_0_gdn_0_cond_true_197997+
'analysis_layer_0_gdn_0_cond_placeholder
(
$analysis_layer_0_gdn_0_cond_identity
?
!analysis/layer_0/gdn_0/cond/ConstConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2#
!analysis/layer_0/gdn_0/cond/Const?
$analysis/layer_0/gdn_0/cond/IdentityIdentity*analysis/layer_0/gdn_0/cond/Const:output:0*
T0
*
_output_shapes
: 2&
$analysis/layer_0/gdn_0/cond/Identity"U
$analysis_layer_0_gdn_0_cond_identity-analysis/layer_0/gdn_0/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
*layer_1_gdn_1_cond_1_cond_cond_true_1985879
5layer_1_gdn_1_cond_1_cond_cond_square_layer_1_biasadd.
*layer_1_gdn_1_cond_1_cond_cond_placeholder+
'layer_1_gdn_1_cond_1_cond_cond_identity?
%layer_1/gdn_1/cond_1/cond/cond/SquareSquare5layer_1_gdn_1_cond_1_cond_cond_square_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2'
%layer_1/gdn_1/cond_1/cond/cond/Square?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity)layer_1/gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity"[
'layer_1_gdn_1_cond_1_cond_cond_identity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_198163K
Ganalysis_layer_1_gdn_1_cond_1_cond_cond_square_analysis_layer_1_biasadd7
3analysis_layer_1_gdn_1_cond_1_cond_cond_placeholder4
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
.analysis/layer_1/gdn_1/cond_1/cond/cond/SquareSquareGanalysis_layer_1_gdn_1_cond_1_cond_cond_square_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.analysis/layer_1/gdn_1/cond_1/cond/cond/Square?
0analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity2analysis/layer_1/gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"m
0analysis_layer_1_gdn_1_cond_1_cond_cond_identity9analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
%layer_2_gdn_2_cond_1_cond_true_1987131
-layer_2_gdn_2_cond_1_cond_abs_layer_2_biasadd)
%layer_2_gdn_2_cond_1_cond_placeholder&
"layer_2_gdn_2_cond_1_cond_identity?
layer_2/gdn_2/cond_1/cond/AbsAbs-layer_2_gdn_2_cond_1_cond_abs_layer_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
layer_2/gdn_2/cond_1/cond/Abs?
"layer_2/gdn_2/cond_1/cond/IdentityIdentity!layer_2/gdn_2/cond_1/cond/Abs:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_2/gdn_2/cond_1/cond/Identity"Q
"layer_2_gdn_2_cond_1_cond_identity+layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
w
layer_1_gdn_1_cond_false_1985583
/layer_1_gdn_1_cond_identity_layer_1_gdn_1_equal

layer_1_gdn_1_cond_identity
?
layer_1/gdn_1/cond/IdentityIdentity/layer_1_gdn_1_cond_identity_layer_1_gdn_1_equal*
T0
*
_output_shapes
: 2
layer_1/gdn_1/cond/Identity"C
layer_1_gdn_1_cond_identity$layer_1/gdn_1/cond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
gdn_2_cond_2_cond_true_196540(
$gdn_2_cond_2_cond_sqrt_gdn_2_biasadd!
gdn_2_cond_2_cond_placeholder
gdn_2_cond_2_cond_identity?
gdn_2/cond_2/cond/SqrtSqrt$gdn_2_cond_2_cond_sqrt_gdn_2_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Sqrt?
gdn_2/cond_2/cond/IdentityIdentitygdn_2/cond_2/cond/Sqrt:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/cond/Identity"A
gdn_2_cond_2_cond_identity#gdn_2/cond_2/cond/Identity:output:0*(
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
"gdn_1_cond_1_cond_cond_true_199457)
%gdn_1_cond_1_cond_cond_square_biasadd&
"gdn_1_cond_1_cond_cond_placeholder#
gdn_1_cond_1_cond_cond_identity?
gdn_1/cond_1/cond/cond/SquareSquare%gdn_1_cond_1_cond_cond_square_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/cond/cond/Square?
gdn_1/cond_1/cond/cond/IdentityIdentity!gdn_1/cond_1/cond/cond/Square:y:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_1/cond_1/cond/cond/Identity"K
gdn_1_cond_1_cond_cond_identity(gdn_1/cond_1/cond/cond/Identity:output:0*(
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
/analysis_layer_2_gdn_2_cond_1_cond_false_198290D
@analysis_layer_2_gdn_2_cond_1_cond_cond_analysis_layer_2_biasadd.
*analysis_layer_2_gdn_2_cond_1_cond_equal_x/
+analysis_layer_2_gdn_2_cond_1_cond_identity?
$analysis/layer_2/gdn_2/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_2/gdn_2/cond_1/cond/x?
(analysis/layer_2/gdn_2/cond_1/cond/EqualEqual*analysis_layer_2_gdn_2_cond_1_cond_equal_x-analysis/layer_2/gdn_2/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_2/gdn_2/cond_1/cond/Equal?
'analysis/layer_2/gdn_2/cond_1/cond/condStatelessIf,analysis/layer_2/gdn_2/cond_1/cond/Equal:z:0@analysis_layer_2_gdn_2_cond_1_cond_cond_analysis_layer_2_biasadd*analysis_layer_2_gdn_2_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_198300*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_1982992)
'analysis/layer_2/gdn_2/cond_1/cond/cond?
0analysis/layer_2/gdn_2/cond_1/cond/cond/IdentityIdentity0analysis/layer_2/gdn_2/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_2/gdn_2/cond_1/cond/cond/Identity?
+analysis/layer_2/gdn_2/cond_1/cond/IdentityIdentity9analysis/layer_2/gdn_2/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_2/gdn_2/cond_1/cond/Identity"c
+analysis_layer_2_gdn_2_cond_1_cond_identity4analysis/layer_2/gdn_2/cond_1/cond/Identity:output:0*(
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
<encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_false_195842X
Tencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_pow_encoder_analysis_layer_1_biasadd9
5encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y<
8encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_identity?
3encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/powPowTencoder_analysis_layer_1_gdn_1_cond_1_cond_cond_pow_encoder_analysis_layer_1_biasadd5encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/pow?
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity7encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2:
8encoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity"}
8encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_identityAencoder/analysis/layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
7encoder_analysis_layer_1_gdn_1_cond_2_cond_false_195915Y
Uencoder_analysis_layer_1_gdn_1_cond_2_cond_pow_encoder_analysis_layer_1_gdn_1_biasadd4
0encoder_analysis_layer_1_gdn_1_cond_2_cond_pow_y7
3encoder_analysis_layer_1_gdn_1_cond_2_cond_identity?
.encoder/analysis/layer_1/gdn_1/cond_2/cond/powPowUencoder_analysis_layer_1_gdn_1_cond_2_cond_pow_encoder_analysis_layer_1_gdn_1_biasadd0encoder_analysis_layer_1_gdn_1_cond_2_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_2/cond/pow?
3encoder/analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity2encoder/analysis/layer_1/gdn_1/cond_2/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity"s
3encoder_analysis_layer_1_gdn_1_cond_2_cond_identity<encoder/analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?N
?
C__inference_layer_0_layer_call_and_return_conditional_losses_196233

inputs
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y
gdn_0_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
Conv2DConv2Dinputslayer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_0/x?
gdn_0/EqualEqualgdn_0_equal_xgdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/Equal?

gdn_0/condStatelessIfgdn_0/Equal:z:0gdn_0/Equal:z:0*
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
 **
else_branchR
gdn_0_cond_false_196110*
output_shapes
: *)
then_branchR
gdn_0_cond_true_1961092

gdn_0/condl
gdn_0/cond/IdentityIdentitygdn_0/cond:output:0*
T0
*
_output_shapes
: 2
gdn_0/cond/Identity?
gdn_0/cond_1StatelessIfgdn_0/cond/Identity:output:0BiasAdd:output:0gdn_0_equal_x*
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
 *,
else_branchR
gdn_0_cond_1_false_196121*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_1_true_1961202
gdn_0/cond_1?
gdn_0/cond_1/IdentityIdentitygdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196166*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196176*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_0/Reshape/shape?
gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:0gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_0/Reshape?
gdn_0/convolutionConv2Dgdn_0/cond_1/Identity:output:0gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196190*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
gdn_0/BiasAddBiasAddgdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/BiasAdd[
	gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_0/x_1?
gdn_0/Equal_1Equalgdn_0_equal_1_xgdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_0/Equal_1?
gdn_0/cond_2StatelessIfgdn_0/Equal_1:z:0gdn_0/BiasAdd:output:0gdn_0_equal_1_x*
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
 *,
else_branchR
gdn_0_cond_2_false_196204*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_2_true_1962032
gdn_0/cond_2?
gdn_0/cond_2/IdentityIdentitygdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity?
gdn_0/truedivRealDivBiasAdd:output:0gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/truediv?
IdentityIdentitygdn_0/truediv:z:0^BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*^
_input_shapesM
K:+???????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
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
??
?
!__inference__wrapped_model_196081
input_1
layer_0_kernel_matmul_a@
-layer_0_kernel_matmul_readvariableop_resource:	?G
8encoder_analysis_layer_0_biasadd_readvariableop_resource:	?*
&encoder_analysis_layer_0_gdn_0_equal_xK
7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource:
??)
%layer_0_gdn_0_gamma_lower_bound_bound
layer_0_gdn_0_gamma_sub_yE
6layer_0_gdn_0_beta_lower_bound_readvariableop_resource:	?(
$layer_0_gdn_0_beta_lower_bound_bound
layer_0_gdn_0_beta_sub_y,
(encoder_analysis_layer_0_gdn_0_equal_1_x
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??G
8encoder_analysis_layer_1_biasadd_readvariableop_resource:	?*
&encoder_analysis_layer_1_gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y,
(encoder_analysis_layer_1_gdn_1_equal_1_x
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??G
8encoder_analysis_layer_2_biasadd_readvariableop_resource:	?*
&encoder_analysis_layer_2_gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y,
(encoder_analysis_layer_2_gdn_2_equal_1_x
layer_3_kernel_matmul_aA
-layer_3_kernel_matmul_readvariableop_resource:
??G
8encoder_analysis_layer_3_biasadd_readvariableop_resource:	?
identity??/encoder/analysis/layer_0/BiasAdd/ReadVariableOp?/encoder/analysis/layer_1/BiasAdd/ReadVariableOp?/encoder/analysis/layer_2/BiasAdd/ReadVariableOp?/encoder/analysis/layer_3/BiasAdd/ReadVariableOp?-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?$layer_0/kernel/MatMul/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?$layer_3/kernel/MatMul/ReadVariableOp?
!encoder/analysis/lambda/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  C2#
!encoder/analysis/lambda/truediv/y?
encoder/analysis/lambda/truedivRealDivinput_1*encoder/analysis/lambda/truediv/y:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2!
encoder/analysis/lambda/truediv?
$layer_0/kernel/MatMul/ReadVariableOpReadVariableOp-layer_0_kernel_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$layer_0/kernel/MatMul/ReadVariableOp?
layer_0/kernel/MatMulMatMullayer_0_kernel_matmul_a,layer_0/kernel/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
layer_0/kernel/MatMul?
layer_0/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"         ?   2
layer_0/kernel/Reshape/shape?
layer_0/kernel/ReshapeReshapelayer_0/kernel/MatMul:product:0%layer_0/kernel/Reshape/shape:output:0*
T0*'
_output_shapes
:?2
layer_0/kernel/Reshape?
encoder/analysis/layer_0/Conv2DConv2D#encoder/analysis/lambda/truediv:z:0layer_0/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2!
encoder/analysis/layer_0/Conv2D?
/encoder/analysis/layer_0/BiasAdd/ReadVariableOpReadVariableOp8encoder_analysis_layer_0_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/encoder/analysis/layer_0/BiasAdd/ReadVariableOp?
 encoder/analysis/layer_0/BiasAddBiasAdd(encoder/analysis/layer_0/Conv2D:output:07encoder/analysis/layer_0/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 encoder/analysis/layer_0/BiasAdd?
 encoder/analysis/layer_0/gdn_0/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 encoder/analysis/layer_0/gdn_0/x?
$encoder/analysis/layer_0/gdn_0/EqualEqual&encoder_analysis_layer_0_gdn_0_equal_x)encoder/analysis/layer_0/gdn_0/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$encoder/analysis/layer_0/gdn_0/Equal?
#encoder/analysis/layer_0/gdn_0/condStatelessIf(encoder/analysis/layer_0/gdn_0/Equal:z:0(encoder/analysis/layer_0/gdn_0/Equal:z:0*
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
0encoder_analysis_layer_0_gdn_0_cond_false_195676*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_0_gdn_0_cond_true_1956752%
#encoder/analysis/layer_0/gdn_0/cond?
,encoder/analysis/layer_0/gdn_0/cond/IdentityIdentity,encoder/analysis/layer_0/gdn_0/cond:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_0/gdn_0/cond/Identity?
%encoder/analysis/layer_0/gdn_0/cond_1StatelessIf5encoder/analysis/layer_0/gdn_0/cond/Identity:output:0)encoder/analysis/layer_0/BiasAdd:output:0&encoder_analysis_layer_0_gdn_0_equal_x*
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
2encoder_analysis_layer_0_gdn_0_cond_1_false_195687*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_0_gdn_0_cond_1_true_1956862'
%encoder/analysis/layer_0/gdn_0/cond_1?
.encoder/analysis/layer_0/gdn_0/cond_1/IdentityIdentity.encoder/analysis/layer_0/gdn_0/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_1/Identity?
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp?
layer_0/gdn_0/gamma/lower_boundMaximum6layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_0/gdn_0/gamma/lower_bound?
(layer_0/gdn_0/gamma/lower_bound/IdentityIdentity#layer_0/gdn_0/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_0/gdn_0/gamma/lower_bound/Identity?
)layer_0/gdn_0/gamma/lower_bound/IdentityN	IdentityN#layer_0/gdn_0/gamma/lower_bound:z:06layer_0/gdn_0/gamma/lower_bound/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195732*.
_output_shapes
:
??:
??: 2+
)layer_0/gdn_0/gamma/lower_bound/IdentityN?
layer_0/gdn_0/gamma/SquareSquare2layer_0/gdn_0/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square?
layer_0/gdn_0/gamma/subSublayer_0/gdn_0/gamma/Square:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub?
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_0_gdn_0_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp?
!layer_0/gdn_0/gamma/lower_bound_1Maximum8layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_0/gdn_0/gamma/lower_bound_1?
*layer_0/gdn_0/gamma/lower_bound_1/IdentityIdentity%layer_0/gdn_0/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_0/gdn_0/gamma/lower_bound_1/Identity?
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN	IdentityN%layer_0/gdn_0/gamma/lower_bound_1:z:08layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp:value:0%layer_0_gdn_0_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195742*.
_output_shapes
:
??:
??: 2-
+layer_0/gdn_0/gamma/lower_bound_1/IdentityN?
layer_0/gdn_0/gamma/Square_1Square4layer_0/gdn_0/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/Square_1?
layer_0/gdn_0/gamma/sub_1Sub layer_0/gdn_0/gamma/Square_1:y:0layer_0_gdn_0_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_0/gdn_0/gamma/sub_1?
,encoder/analysis/layer_0/gdn_0/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,encoder/analysis/layer_0/gdn_0/Reshape/shape?
&encoder/analysis/layer_0/gdn_0/ReshapeReshapelayer_0/gdn_0/gamma/sub_1:z:05encoder/analysis/layer_0/gdn_0/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&encoder/analysis/layer_0/gdn_0/Reshape?
*encoder/analysis/layer_0/gdn_0/convolutionConv2D7encoder/analysis/layer_0/gdn_0/cond_1/Identity:output:0/encoder/analysis/layer_0/gdn_0/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*encoder/analysis/layer_0/gdn_0/convolution?
-layer_0/gdn_0/beta/lower_bound/ReadVariableOpReadVariableOp6layer_0_gdn_0_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp?
layer_0/gdn_0/beta/lower_boundMaximum5layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_0/gdn_0/beta/lower_bound?
'layer_0/gdn_0/beta/lower_bound/IdentityIdentity"layer_0/gdn_0/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_0/gdn_0/beta/lower_bound/Identity?
(layer_0/gdn_0/beta/lower_bound/IdentityN	IdentityN"layer_0/gdn_0/beta/lower_bound:z:05layer_0/gdn_0/beta/lower_bound/ReadVariableOp:value:0$layer_0_gdn_0_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195756*$
_output_shapes
:?:?: 2*
(layer_0/gdn_0/beta/lower_bound/IdentityN?
layer_0/gdn_0/beta/SquareSquare1layer_0/gdn_0/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/Square?
layer_0/gdn_0/beta/subSublayer_0/gdn_0/beta/Square:y:0layer_0_gdn_0_beta_sub_y*
T0*
_output_shapes	
:?2
layer_0/gdn_0/beta/sub?
&encoder/analysis/layer_0/gdn_0/BiasAddBiasAdd3encoder/analysis/layer_0/gdn_0/convolution:output:0layer_0/gdn_0/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_0/gdn_0/BiasAdd?
"encoder/analysis/layer_0/gdn_0/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"encoder/analysis/layer_0/gdn_0/x_1?
&encoder/analysis/layer_0/gdn_0/Equal_1Equal(encoder_analysis_layer_0_gdn_0_equal_1_x+encoder/analysis/layer_0/gdn_0/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&encoder/analysis/layer_0/gdn_0/Equal_1?
%encoder/analysis/layer_0/gdn_0/cond_2StatelessIf*encoder/analysis/layer_0/gdn_0/Equal_1:z:0/encoder/analysis/layer_0/gdn_0/BiasAdd:output:0(encoder_analysis_layer_0_gdn_0_equal_1_x*
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
2encoder_analysis_layer_0_gdn_0_cond_2_false_195770*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_0_gdn_0_cond_2_true_1957692'
%encoder/analysis/layer_0/gdn_0/cond_2?
.encoder/analysis/layer_0/gdn_0/cond_2/IdentityIdentity.encoder/analysis/layer_0/gdn_0/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_2/Identity?
&encoder/analysis/layer_0/gdn_0/truedivRealDiv)encoder/analysis/layer_0/BiasAdd:output:07encoder/analysis/layer_0/gdn_0/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_0/gdn_0/truediv?
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
layer_1/kernel/Reshape?
encoder/analysis/layer_1/Conv2DConv2D*encoder/analysis/layer_0/gdn_0/truediv:z:0layer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2!
encoder/analysis/layer_1/Conv2D?
/encoder/analysis/layer_1/BiasAdd/ReadVariableOpReadVariableOp8encoder_analysis_layer_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/encoder/analysis/layer_1/BiasAdd/ReadVariableOp?
 encoder/analysis/layer_1/BiasAddBiasAdd(encoder/analysis/layer_1/Conv2D:output:07encoder/analysis/layer_1/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 encoder/analysis/layer_1/BiasAdd?
 encoder/analysis/layer_1/gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 encoder/analysis/layer_1/gdn_1/x?
$encoder/analysis/layer_1/gdn_1/EqualEqual&encoder_analysis_layer_1_gdn_1_equal_x)encoder/analysis/layer_1/gdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$encoder/analysis/layer_1/gdn_1/Equal?
#encoder/analysis/layer_1/gdn_1/condStatelessIf(encoder/analysis/layer_1/gdn_1/Equal:z:0(encoder/analysis/layer_1/gdn_1/Equal:z:0*
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
0encoder_analysis_layer_1_gdn_1_cond_false_195812*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_1_gdn_1_cond_true_1958112%
#encoder/analysis/layer_1/gdn_1/cond?
,encoder/analysis/layer_1/gdn_1/cond/IdentityIdentity,encoder/analysis/layer_1/gdn_1/cond:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_1/gdn_1/cond/Identity?
%encoder/analysis/layer_1/gdn_1/cond_1StatelessIf5encoder/analysis/layer_1/gdn_1/cond/Identity:output:0)encoder/analysis/layer_1/BiasAdd:output:0&encoder_analysis_layer_1_gdn_1_equal_x*
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
2encoder_analysis_layer_1_gdn_1_cond_1_false_195823*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_1_gdn_1_cond_1_true_1958222'
%encoder/analysis/layer_1/gdn_1/cond_1?
.encoder/analysis/layer_1/gdn_1/cond_1/IdentityIdentity.encoder/analysis/layer_1/gdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195868*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195878*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
,encoder/analysis/layer_1/gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,encoder/analysis/layer_1/gdn_1/Reshape/shape?
&encoder/analysis/layer_1/gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:05encoder/analysis/layer_1/gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&encoder/analysis/layer_1/gdn_1/Reshape?
*encoder/analysis/layer_1/gdn_1/convolutionConv2D7encoder/analysis/layer_1/gdn_1/cond_1/Identity:output:0/encoder/analysis/layer_1/gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*encoder/analysis/layer_1/gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-195892*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
&encoder/analysis/layer_1/gdn_1/BiasAddBiasAdd3encoder/analysis/layer_1/gdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_1/gdn_1/BiasAdd?
"encoder/analysis/layer_1/gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"encoder/analysis/layer_1/gdn_1/x_1?
&encoder/analysis/layer_1/gdn_1/Equal_1Equal(encoder_analysis_layer_1_gdn_1_equal_1_x+encoder/analysis/layer_1/gdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&encoder/analysis/layer_1/gdn_1/Equal_1?
%encoder/analysis/layer_1/gdn_1/cond_2StatelessIf*encoder/analysis/layer_1/gdn_1/Equal_1:z:0/encoder/analysis/layer_1/gdn_1/BiasAdd:output:0(encoder_analysis_layer_1_gdn_1_equal_1_x*
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
2encoder_analysis_layer_1_gdn_1_cond_2_false_195906*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_1_gdn_1_cond_2_true_1959052'
%encoder/analysis/layer_1/gdn_1/cond_2?
.encoder/analysis/layer_1/gdn_1/cond_2/IdentityIdentity.encoder/analysis/layer_1/gdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_2/Identity?
&encoder/analysis/layer_1/gdn_1/truedivRealDiv)encoder/analysis/layer_1/BiasAdd:output:07encoder/analysis/layer_1/gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_1/gdn_1/truediv?
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
layer_2/kernel/Reshape?
encoder/analysis/layer_2/Conv2DConv2D*encoder/analysis/layer_1/gdn_1/truediv:z:0layer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2!
encoder/analysis/layer_2/Conv2D?
/encoder/analysis/layer_2/BiasAdd/ReadVariableOpReadVariableOp8encoder_analysis_layer_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/encoder/analysis/layer_2/BiasAdd/ReadVariableOp?
 encoder/analysis/layer_2/BiasAddBiasAdd(encoder/analysis/layer_2/Conv2D:output:07encoder/analysis/layer_2/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 encoder/analysis/layer_2/BiasAdd?
 encoder/analysis/layer_2/gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 encoder/analysis/layer_2/gdn_2/x?
$encoder/analysis/layer_2/gdn_2/EqualEqual&encoder_analysis_layer_2_gdn_2_equal_x)encoder/analysis/layer_2/gdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2&
$encoder/analysis/layer_2/gdn_2/Equal?
#encoder/analysis/layer_2/gdn_2/condStatelessIf(encoder/analysis/layer_2/gdn_2/Equal:z:0(encoder/analysis/layer_2/gdn_2/Equal:z:0*
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
0encoder_analysis_layer_2_gdn_2_cond_false_195948*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_2_gdn_2_cond_true_1959472%
#encoder/analysis/layer_2/gdn_2/cond?
,encoder/analysis/layer_2/gdn_2/cond/IdentityIdentity,encoder/analysis/layer_2/gdn_2/cond:output:0*
T0
*
_output_shapes
: 2.
,encoder/analysis/layer_2/gdn_2/cond/Identity?
%encoder/analysis/layer_2/gdn_2/cond_1StatelessIf5encoder/analysis/layer_2/gdn_2/cond/Identity:output:0)encoder/analysis/layer_2/BiasAdd:output:0&encoder_analysis_layer_2_gdn_2_equal_x*
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
2encoder_analysis_layer_2_gdn_2_cond_1_false_195959*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_2_gdn_2_cond_1_true_1959582'
%encoder/analysis/layer_2/gdn_2/cond_1?
.encoder/analysis/layer_2/gdn_2/cond_1/IdentityIdentity.encoder/analysis/layer_2/gdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196004*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196014*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
,encoder/analysis/layer_2/gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2.
,encoder/analysis/layer_2/gdn_2/Reshape/shape?
&encoder/analysis/layer_2/gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:05encoder/analysis/layer_2/gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2(
&encoder/analysis/layer_2/gdn_2/Reshape?
*encoder/analysis/layer_2/gdn_2/convolutionConv2D7encoder/analysis/layer_2/gdn_2/cond_1/Identity:output:0/encoder/analysis/layer_2/gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2,
*encoder/analysis/layer_2/gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196028*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
&encoder/analysis/layer_2/gdn_2/BiasAddBiasAdd3encoder/analysis/layer_2/gdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_2/gdn_2/BiasAdd?
"encoder/analysis/layer_2/gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"encoder/analysis/layer_2/gdn_2/x_1?
&encoder/analysis/layer_2/gdn_2/Equal_1Equal(encoder_analysis_layer_2_gdn_2_equal_1_x+encoder/analysis/layer_2/gdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2(
&encoder/analysis/layer_2/gdn_2/Equal_1?
%encoder/analysis/layer_2/gdn_2/cond_2StatelessIf*encoder/analysis/layer_2/gdn_2/Equal_1:z:0/encoder/analysis/layer_2/gdn_2/BiasAdd:output:0(encoder_analysis_layer_2_gdn_2_equal_1_x*
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
2encoder_analysis_layer_2_gdn_2_cond_2_false_196042*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_2_gdn_2_cond_2_true_1960412'
%encoder/analysis/layer_2/gdn_2/cond_2?
.encoder/analysis/layer_2/gdn_2/cond_2/IdentityIdentity.encoder/analysis/layer_2/gdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_2/gdn_2/cond_2/Identity?
&encoder/analysis/layer_2/gdn_2/truedivRealDiv)encoder/analysis/layer_2/BiasAdd:output:07encoder/analysis/layer_2/gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&encoder/analysis/layer_2/gdn_2/truediv?
$layer_3/kernel/MatMul/ReadVariableOpReadVariableOp-layer_3_kernel_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02&
$layer_3/kernel/MatMul/ReadVariableOp?
layer_3/kernel/MatMulMatMullayer_3_kernel_matmul_a,layer_3/kernel/MatMul/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2
layer_3/kernel/MatMul?
layer_3/kernel/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
layer_3/kernel/Reshape/shape?
layer_3/kernel/ReshapeReshapelayer_3/kernel/MatMul:product:0%layer_3/kernel/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
layer_3/kernel/Reshape?
encoder/analysis/layer_3/Conv2DConv2D*encoder/analysis/layer_2/gdn_2/truediv:z:0layer_3/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2!
encoder/analysis/layer_3/Conv2D?
/encoder/analysis/layer_3/BiasAdd/ReadVariableOpReadVariableOp8encoder_analysis_layer_3_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/encoder/analysis/layer_3/BiasAdd/ReadVariableOp?
 encoder/analysis/layer_3/BiasAddBiasAdd(encoder/analysis/layer_3/Conv2D:output:07encoder/analysis/layer_3/BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2"
 encoder/analysis/layer_3/BiasAdd?
IdentityIdentity)encoder/analysis/layer_3/BiasAdd:output:00^encoder/analysis/layer_0/BiasAdd/ReadVariableOp0^encoder/analysis/layer_1/BiasAdd/ReadVariableOp0^encoder/analysis/layer_2/BiasAdd/ReadVariableOp0^encoder/analysis/layer_3/BiasAdd/ReadVariableOp.^layer_0/gdn_0/beta/lower_bound/ReadVariableOp/^layer_0/gdn_0/gamma/lower_bound/ReadVariableOp1^layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp%^layer_0/kernel/MatMul/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp%^layer_3/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:+???????????????????????????:: : : : : : : : : : :: : : : : : : : : : :: : : : : : : : : : :: : 2b
/encoder/analysis/layer_0/BiasAdd/ReadVariableOp/encoder/analysis/layer_0/BiasAdd/ReadVariableOp2b
/encoder/analysis/layer_1/BiasAdd/ReadVariableOp/encoder/analysis/layer_1/BiasAdd/ReadVariableOp2b
/encoder/analysis/layer_2/BiasAdd/ReadVariableOp/encoder/analysis/layer_2/BiasAdd/ReadVariableOp2b
/encoder/analysis/layer_3/BiasAdd/ReadVariableOp/encoder/analysis/layer_3/BiasAdd/ReadVariableOp2^
-layer_0/gdn_0/beta/lower_bound/ReadVariableOp-layer_0/gdn_0/beta/lower_bound/ReadVariableOp2`
.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp.layer_0/gdn_0/gamma/lower_bound/ReadVariableOp2d
0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp0layer_0/gdn_0/gamma/lower_bound_1/ReadVariableOp2L
$layer_0/kernel/MatMul/ReadVariableOp$layer_0/kernel/MatMul/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
$layer_1/kernel/MatMul/ReadVariableOp$layer_1/kernel/MatMul/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
$layer_2/kernel/MatMul/ReadVariableOp$layer_2/kernel/MatMul/ReadVariableOp2L
$layer_3/kernel/MatMul/ReadVariableOp$layer_3/kernel/MatMul/ReadVariableOp:j f
A
_output_shapes/
-:+???????????????????????????
!
_user_specified_name	input_1:$ 

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
2encoder_analysis_layer_0_gdn_0_cond_1_false_195687O
Kencoder_analysis_layer_0_gdn_0_cond_1_cond_encoder_analysis_layer_0_biasadd1
-encoder_analysis_layer_0_gdn_0_cond_1_equal_x2
.encoder_analysis_layer_0_gdn_0_cond_1_identity?
'encoder/analysis/layer_0/gdn_0/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'encoder/analysis/layer_0/gdn_0/cond_1/x?
+encoder/analysis/layer_0/gdn_0/cond_1/EqualEqual-encoder_analysis_layer_0_gdn_0_cond_1_equal_x0encoder/analysis/layer_0/gdn_0/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2-
+encoder/analysis/layer_0/gdn_0/cond_1/Equal?
*encoder/analysis/layer_0/gdn_0/cond_1/condStatelessIf/encoder/analysis/layer_0/gdn_0/cond_1/Equal:z:0Kencoder_analysis_layer_0_gdn_0_cond_1_cond_encoder_analysis_layer_0_biasadd-encoder_analysis_layer_0_gdn_0_cond_1_equal_x*
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
7encoder_analysis_layer_0_gdn_0_cond_1_cond_false_195696*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_0_gdn_0_cond_1_cond_true_1956952,
*encoder/analysis/layer_0/gdn_0/cond_1/cond?
3encoder/analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity3encoder/analysis/layer_0/gdn_0/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????25
3encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity?
.encoder/analysis/layer_0/gdn_0/cond_1/IdentityIdentity<encoder/analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_1/Identity"i
.encoder_analysis_layer_0_gdn_0_cond_1_identity7encoder/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
*analysis_layer_1_gdn_1_cond_2_false_198228E
Aanalysis_layer_1_gdn_1_cond_2_cond_analysis_layer_1_gdn_1_biasadd)
%analysis_layer_1_gdn_1_cond_2_equal_x*
&analysis_layer_1_gdn_1_cond_2_identity?
analysis/layer_1/gdn_1/cond_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
analysis/layer_1/gdn_1/cond_2/x?
#analysis/layer_1/gdn_1/cond_2/EqualEqual%analysis_layer_1_gdn_1_cond_2_equal_x(analysis/layer_1/gdn_1/cond_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2%
#analysis/layer_1/gdn_1/cond_2/Equal?
"analysis/layer_1/gdn_1/cond_2/condStatelessIf'analysis/layer_1/gdn_1/cond_2/Equal:z:0Aanalysis_layer_1_gdn_1_cond_2_cond_analysis_layer_1_gdn_1_biasadd%analysis_layer_1_gdn_1_cond_2_equal_x*
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
 *B
else_branch3R1
/analysis_layer_1_gdn_1_cond_2_cond_false_198237*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_2_cond_true_1982362$
"analysis/layer_1/gdn_1/cond_2/cond?
+analysis/layer_1/gdn_1/cond_2/cond/IdentityIdentity+analysis/layer_1/gdn_1/cond_2/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_1/gdn_1/cond_2/cond/Identity?
&analysis/layer_1/gdn_1/cond_2/IdentityIdentity4analysis/layer_1/gdn_1/cond_2/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_1/gdn_1/cond_2/Identity"Y
&analysis_layer_1_gdn_1_cond_2_identity/analysis/layer_1/gdn_1/cond_2/Identity:output:0*(
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
#gdn_0_cond_1_cond_cond_false_199314&
"gdn_0_cond_1_cond_cond_pow_biasadd 
gdn_0_cond_1_cond_cond_pow_y#
gdn_0_cond_1_cond_cond_identity?
gdn_0/cond_1/cond/cond/powPow"gdn_0_cond_1_cond_cond_pow_biasaddgdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_1/cond/cond/pow?
gdn_0/cond_1/cond/cond/IdentityIdentitygdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2!
gdn_0/cond_1/cond/cond/Identity"K
gdn_0_cond_1_cond_cond_identity(gdn_0/cond_1/cond/cond/Identity:output:0*(
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
+layer_0_gdn_0_cond_1_cond_cond_false_1984526
2layer_0_gdn_0_cond_1_cond_cond_pow_layer_0_biasadd(
$layer_0_gdn_0_cond_1_cond_cond_pow_y+
'layer_0_gdn_0_cond_1_cond_cond_identity?
"layer_0/gdn_0/cond_1/cond/cond/powPow2layer_0_gdn_0_cond_1_cond_cond_pow_layer_0_biasadd$layer_0_gdn_0_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_0/gdn_0/cond_1/cond/cond/pow?
'layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity&layer_0/gdn_0/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_0/gdn_0/cond_1/cond/cond/Identity"[
'layer_0_gdn_0_cond_1_cond_cond_identity0layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*(
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
+layer_1_gdn_1_cond_1_cond_cond_false_1985886
2layer_1_gdn_1_cond_1_cond_cond_pow_layer_1_biasadd(
$layer_1_gdn_1_cond_1_cond_cond_pow_y+
'layer_1_gdn_1_cond_1_cond_cond_identity?
"layer_1/gdn_1/cond_1/cond/cond/powPow2layer_1_gdn_1_cond_1_cond_cond_pow_layer_1_biasadd$layer_1_gdn_1_cond_1_cond_cond_pow_y*
T0*B
_output_shapes0
.:,????????????????????????????2$
"layer_1/gdn_1/cond_1/cond/cond/pow?
'layer_1/gdn_1/cond_1/cond/cond/IdentityIdentity&layer_1/gdn_1/cond_1/cond/cond/pow:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2)
'layer_1/gdn_1/cond_1/cond/cond/Identity"[
'layer_1_gdn_1_cond_1_cond_cond_identity0layer_1/gdn_1/cond_1/cond/cond/Identity:output:0*(
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
o
gdn_2_cond_1_false_196449
gdn_2_cond_1_cond_biasadd
gdn_2_cond_1_equal_x
gdn_2_cond_1_identitye
gdn_2/cond_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gdn_2/cond_1/x?
gdn_2/cond_1/EqualEqualgdn_2_cond_1_equal_xgdn_2/cond_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/cond_1/Equal?
gdn_2/cond_1/condStatelessIfgdn_2/cond_1/Equal:z:0gdn_2_cond_1_cond_biasaddgdn_2_cond_1_equal_x*
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
 *1
else_branch"R 
gdn_2_cond_1_cond_false_196458*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_1_cond_true_1964572
gdn_2/cond_1/cond?
gdn_2/cond_1/cond/IdentityIdentitygdn_2/cond_1/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/cond/Identity?
gdn_2/cond_1/IdentityIdentity#gdn_2/cond_1/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity"7
gdn_2_cond_1_identitygdn_2/cond_1/Identity:output:0*(
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
1encoder_analysis_layer_0_gdn_0_cond_2_true_195769Y
Uencoder_analysis_layer_0_gdn_0_cond_2_identity_encoder_analysis_layer_0_gdn_0_biasadd5
1encoder_analysis_layer_0_gdn_0_cond_2_placeholder2
.encoder_analysis_layer_0_gdn_0_cond_2_identity?
.encoder/analysis/layer_0/gdn_0/cond_2/IdentityIdentityUencoder_analysis_layer_0_gdn_0_cond_2_identity_encoder_analysis_layer_0_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_0/gdn_0/cond_2/Identity"i
.encoder_analysis_layer_0_gdn_0_cond_2_identity7encoder/analysis/layer_0/gdn_0/cond_2/Identity:output:0*(
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
/analysis_layer_0_gdn_0_cond_1_cond_false_198018D
@analysis_layer_0_gdn_0_cond_1_cond_cond_analysis_layer_0_biasadd.
*analysis_layer_0_gdn_0_cond_1_cond_equal_x/
+analysis_layer_0_gdn_0_cond_1_cond_identity?
$analysis/layer_0/gdn_0/cond_1/cond/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @2&
$analysis/layer_0/gdn_0/cond_1/cond/x?
(analysis/layer_0/gdn_0/cond_1/cond/EqualEqual*analysis_layer_0_gdn_0_cond_1_cond_equal_x-analysis/layer_0/gdn_0/cond_1/cond/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2*
(analysis/layer_0/gdn_0/cond_1/cond/Equal?
'analysis/layer_0/gdn_0/cond_1/cond/condStatelessIf,analysis/layer_0/gdn_0/cond_1/cond/Equal:z:0@analysis_layer_0_gdn_0_cond_1_cond_cond_analysis_layer_0_biasadd*analysis_layer_0_gdn_0_cond_1_cond_equal_x*
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
 *G
else_branch8R6
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_198028*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_1980272)
'analysis/layer_0/gdn_0/cond_1/cond/cond?
0analysis/layer_0/gdn_0/cond_1/cond/cond/IdentityIdentity0analysis/layer_0/gdn_0/cond_1/cond/cond:output:0*
T0*B
_output_shapes0
.:,????????????????????????????22
0analysis/layer_0/gdn_0/cond_1/cond/cond/Identity?
+analysis/layer_0/gdn_0/cond_1/cond/IdentityIdentity9analysis/layer_0/gdn_0/cond_1/cond/cond/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2-
+analysis/layer_0/gdn_0/cond_1/cond/Identity"c
+analysis_layer_0_gdn_0_cond_1_cond_identity4analysis/layer_0/gdn_0/cond_1/cond/Identity:output:0*(
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
1encoder_analysis_layer_1_gdn_1_cond_1_true_195822S
Oencoder_analysis_layer_1_gdn_1_cond_1_identity_encoder_analysis_layer_1_biasadd5
1encoder_analysis_layer_1_gdn_1_cond_1_placeholder2
.encoder_analysis_layer_1_gdn_1_cond_1_identity?
.encoder/analysis/layer_1/gdn_1/cond_1/IdentityIdentityOencoder_analysis_layer_1_gdn_1_cond_1_identity_encoder_analysis_layer_1_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????20
.encoder/analysis/layer_1/gdn_1/cond_1/Identity"i
.encoder_analysis_layer_1_gdn_1_cond_1_identity7encoder/analysis/layer_1/gdn_1/cond_1/Identity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: :H D
B
_output_shapes0
.:,????????????????????????????:

_output_shapes
: 
?N
?
C__inference_layer_2_layer_call_and_return_conditional_losses_199695

inputs
layer_2_kernel_matmul_aA
-layer_2_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
gdn_2_equal_xK
7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource:
??)
%layer_2_gdn_2_gamma_lower_bound_bound
layer_2_gdn_2_gamma_sub_yE
6layer_2_gdn_2_beta_lower_bound_readvariableop_resource:	?(
$layer_2_gdn_2_beta_lower_bound_bound
layer_2_gdn_2_beta_sub_y
gdn_2_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?$layer_2/kernel/MatMul/ReadVariableOp?
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
Conv2DConv2Dinputslayer_2/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_2/x?
gdn_2/EqualEqualgdn_2_equal_xgdn_2/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/Equal?

gdn_2/condStatelessIfgdn_2/Equal:z:0gdn_2/Equal:z:0*
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
 **
else_branchR
gdn_2_cond_false_199572*
output_shapes
: *)
then_branchR
gdn_2_cond_true_1995712

gdn_2/condl
gdn_2/cond/IdentityIdentitygdn_2/cond:output:0*
T0
*
_output_shapes
: 2
gdn_2/cond/Identity?
gdn_2/cond_1StatelessIfgdn_2/cond/Identity:output:0BiasAdd:output:0gdn_2_equal_x*
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
 *,
else_branchR
gdn_2_cond_1_false_199583*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_1_true_1995822
gdn_2/cond_1?
gdn_2/cond_1/IdentityIdentitygdn_2/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_1/Identity?
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp?
layer_2/gdn_2/gamma/lower_boundMaximum6layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_2/gdn_2/gamma/lower_bound?
(layer_2/gdn_2/gamma/lower_bound/IdentityIdentity#layer_2/gdn_2/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_2/gdn_2/gamma/lower_bound/Identity?
)layer_2/gdn_2/gamma/lower_bound/IdentityN	IdentityN#layer_2/gdn_2/gamma/lower_bound:z:06layer_2/gdn_2/gamma/lower_bound/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199628*.
_output_shapes
:
??:
??: 2+
)layer_2/gdn_2/gamma/lower_bound/IdentityN?
layer_2/gdn_2/gamma/SquareSquare2layer_2/gdn_2/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square?
layer_2/gdn_2/gamma/subSublayer_2/gdn_2/gamma/Square:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub?
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_2_gdn_2_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp?
!layer_2/gdn_2/gamma/lower_bound_1Maximum8layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_2/gdn_2/gamma/lower_bound_1?
*layer_2/gdn_2/gamma/lower_bound_1/IdentityIdentity%layer_2/gdn_2/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_2/gdn_2/gamma/lower_bound_1/Identity?
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN	IdentityN%layer_2/gdn_2/gamma/lower_bound_1:z:08layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp:value:0%layer_2_gdn_2_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199638*.
_output_shapes
:
??:
??: 2-
+layer_2/gdn_2/gamma/lower_bound_1/IdentityN?
layer_2/gdn_2/gamma/Square_1Square4layer_2/gdn_2/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/Square_1?
layer_2/gdn_2/gamma/sub_1Sub layer_2/gdn_2/gamma/Square_1:y:0layer_2_gdn_2_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_2/gdn_2/gamma/sub_1?
gdn_2/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_2/Reshape/shape?
gdn_2/ReshapeReshapelayer_2/gdn_2/gamma/sub_1:z:0gdn_2/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_2/Reshape?
gdn_2/convolutionConv2Dgdn_2/cond_1/Identity:output:0gdn_2/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_2/convolution?
-layer_2/gdn_2/beta/lower_bound/ReadVariableOpReadVariableOp6layer_2_gdn_2_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp?
layer_2/gdn_2/beta/lower_boundMaximum5layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_2/gdn_2/beta/lower_bound?
'layer_2/gdn_2/beta/lower_bound/IdentityIdentity"layer_2/gdn_2/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_2/gdn_2/beta/lower_bound/Identity?
(layer_2/gdn_2/beta/lower_bound/IdentityN	IdentityN"layer_2/gdn_2/beta/lower_bound:z:05layer_2/gdn_2/beta/lower_bound/ReadVariableOp:value:0$layer_2_gdn_2_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-199652*$
_output_shapes
:?:?: 2*
(layer_2/gdn_2/beta/lower_bound/IdentityN?
layer_2/gdn_2/beta/SquareSquare1layer_2/gdn_2/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/Square?
layer_2/gdn_2/beta/subSublayer_2/gdn_2/beta/Square:y:0layer_2_gdn_2_beta_sub_y*
T0*
_output_shapes	
:?2
layer_2/gdn_2/beta/sub?
gdn_2/BiasAddBiasAddgdn_2/convolution:output:0layer_2/gdn_2/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/BiasAdd[
	gdn_2/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_2/x_1?
gdn_2/Equal_1Equalgdn_2_equal_1_xgdn_2/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_2/Equal_1?
gdn_2/cond_2StatelessIfgdn_2/Equal_1:z:0gdn_2/BiasAdd:output:0gdn_2_equal_1_x*
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
 *,
else_branchR
gdn_2_cond_2_false_199666*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_2_true_1996652
gdn_2/cond_2?
gdn_2/cond_2/IdentityIdentitygdn_2/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/cond_2/Identity?
gdn_2/truedivRealDivBiasAdd:output:0gdn_2/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_2/truediv?
IdentityIdentitygdn_2/truediv:z:0^BiasAdd/ReadVariableOp.^layer_2/gdn_2/beta/lower_bound/ReadVariableOp/^layer_2/gdn_2/gamma/lower_bound/ReadVariableOp1^layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp%^layer_2/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_2/gdn_2/beta/lower_bound/ReadVariableOp-layer_2/gdn_2/beta/lower_bound/ReadVariableOp2`
.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp.layer_2/gdn_2/gamma/lower_bound/ReadVariableOp2d
0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp0layer_2/gdn_2/gamma/lower_bound_1/ReadVariableOp2L
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
?N
?
C__inference_layer_1_layer_call_and_return_conditional_losses_196397

inputs
layer_1_kernel_matmul_aA
-layer_1_kernel_matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
gdn_1_equal_xK
7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource:
??)
%layer_1_gdn_1_gamma_lower_bound_bound
layer_1_gdn_1_gamma_sub_yE
6layer_1_gdn_1_beta_lower_bound_readvariableop_resource:	?(
$layer_1_gdn_1_beta_lower_bound_bound
layer_1_gdn_1_beta_sub_y
gdn_1_equal_1_x
identity??BiasAdd/ReadVariableOp?-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?$layer_1/kernel/MatMul/ReadVariableOp?
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
Conv2DConv2Dinputslayer_1/kernel/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*!
explicit_paddings

    *
padding
EXPLICIT*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAddW
gdn_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2	
gdn_1/x?
gdn_1/EqualEqualgdn_1_equal_xgdn_1/x:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/Equal?

gdn_1/condStatelessIfgdn_1/Equal:z:0gdn_1/Equal:z:0*
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
 **
else_branchR
gdn_1_cond_false_196274*
output_shapes
: *)
then_branchR
gdn_1_cond_true_1962732

gdn_1/condl
gdn_1/cond/IdentityIdentitygdn_1/cond:output:0*
T0
*
_output_shapes
: 2
gdn_1/cond/Identity?
gdn_1/cond_1StatelessIfgdn_1/cond/Identity:output:0BiasAdd:output:0gdn_1_equal_x*
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
 *,
else_branchR
gdn_1_cond_1_false_196285*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_1_true_1962842
gdn_1/cond_1?
gdn_1/cond_1/IdentityIdentitygdn_1/cond_1:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_1/Identity?
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype020
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp?
layer_1/gdn_1/gamma/lower_boundMaximum6layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2!
layer_1/gdn_1/gamma/lower_bound?
(layer_1/gdn_1/gamma/lower_bound/IdentityIdentity#layer_1/gdn_1/gamma/lower_bound:z:0*
T0* 
_output_shapes
:
??2*
(layer_1/gdn_1/gamma/lower_bound/Identity?
)layer_1/gdn_1/gamma/lower_bound/IdentityN	IdentityN#layer_1/gdn_1/gamma/lower_bound:z:06layer_1/gdn_1/gamma/lower_bound/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196330*.
_output_shapes
:
??:
??: 2+
)layer_1/gdn_1/gamma/lower_bound/IdentityN?
layer_1/gdn_1/gamma/SquareSquare2layer_1/gdn_1/gamma/lower_bound/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square?
layer_1/gdn_1/gamma/subSublayer_1/gdn_1/gamma/Square:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub?
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOpReadVariableOp7layer_1_gdn_1_gamma_lower_bound_readvariableop_resource* 
_output_shapes
:
??*
dtype022
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp?
!layer_1/gdn_1/gamma/lower_bound_1Maximum8layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T0* 
_output_shapes
:
??2#
!layer_1/gdn_1/gamma/lower_bound_1?
*layer_1/gdn_1/gamma/lower_bound_1/IdentityIdentity%layer_1/gdn_1/gamma/lower_bound_1:z:0*
T0* 
_output_shapes
:
??2,
*layer_1/gdn_1/gamma/lower_bound_1/Identity?
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN	IdentityN%layer_1/gdn_1/gamma/lower_bound_1:z:08layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp:value:0%layer_1_gdn_1_gamma_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196340*.
_output_shapes
:
??:
??: 2-
+layer_1/gdn_1/gamma/lower_bound_1/IdentityN?
layer_1/gdn_1/gamma/Square_1Square4layer_1/gdn_1/gamma/lower_bound_1/IdentityN:output:0*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/Square_1?
layer_1/gdn_1/gamma/sub_1Sub layer_1/gdn_1/gamma/Square_1:y:0layer_1_gdn_1_gamma_sub_y*
T0* 
_output_shapes
:
??2
layer_1/gdn_1/gamma/sub_1?
gdn_1/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"      ?   ?   2
gdn_1/Reshape/shape?
gdn_1/ReshapeReshapelayer_1/gdn_1/gamma/sub_1:z:0gdn_1/Reshape/shape:output:0*
T0*(
_output_shapes
:??2
gdn_1/Reshape?
gdn_1/convolutionConv2Dgdn_1/cond_1/Identity:output:0gdn_1/Reshape:output:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
gdn_1/convolution?
-layer_1/gdn_1/beta/lower_bound/ReadVariableOpReadVariableOp6layer_1_gdn_1_beta_lower_bound_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp?
layer_1/gdn_1/beta/lower_boundMaximum5layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T0*
_output_shapes	
:?2 
layer_1/gdn_1/beta/lower_bound?
'layer_1/gdn_1/beta/lower_bound/IdentityIdentity"layer_1/gdn_1/beta/lower_bound:z:0*
T0*
_output_shapes	
:?2)
'layer_1/gdn_1/beta/lower_bound/Identity?
(layer_1/gdn_1/beta/lower_bound/IdentityN	IdentityN"layer_1/gdn_1/beta/lower_bound:z:05layer_1/gdn_1/beta/lower_bound/ReadVariableOp:value:0$layer_1_gdn_1_beta_lower_bound_bound*
T
2*,
_gradient_op_typeCustomGradient-196354*$
_output_shapes
:?:?: 2*
(layer_1/gdn_1/beta/lower_bound/IdentityN?
layer_1/gdn_1/beta/SquareSquare1layer_1/gdn_1/beta/lower_bound/IdentityN:output:0*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/Square?
layer_1/gdn_1/beta/subSublayer_1/gdn_1/beta/Square:y:0layer_1_gdn_1_beta_sub_y*
T0*
_output_shapes	
:?2
layer_1/gdn_1/beta/sub?
gdn_1/BiasAddBiasAddgdn_1/convolution:output:0layer_1/gdn_1/beta/sub:z:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/BiasAdd[
	gdn_1/x_1Const*
_output_shapes
: *
dtype0*
valueB
 *  ??2
	gdn_1/x_1?
gdn_1/Equal_1Equalgdn_1_equal_1_xgdn_1/x_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gdn_1/Equal_1?
gdn_1/cond_2StatelessIfgdn_1/Equal_1:z:0gdn_1/BiasAdd:output:0gdn_1_equal_1_x*
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
 *,
else_branchR
gdn_1_cond_2_false_196368*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_2_true_1963672
gdn_1/cond_2?
gdn_1/cond_2/IdentityIdentitygdn_1/cond_2:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/cond_2/Identity?
gdn_1/truedivRealDivBiasAdd:output:0gdn_1/cond_2/Identity:output:0*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_1/truediv?
IdentityIdentitygdn_1/truediv:z:0^BiasAdd/ReadVariableOp.^layer_1/gdn_1/beta/lower_bound/ReadVariableOp/^layer_1/gdn_1/gamma/lower_bound/ReadVariableOp1^layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp%^layer_1/kernel/MatMul/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*_
_input_shapesN
L:,????????????????????????????:: : : : : : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2^
-layer_1/gdn_1/beta/lower_bound/ReadVariableOp-layer_1/gdn_1/beta/lower_bound/ReadVariableOp2`
.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp.layer_1/gdn_1/gamma/lower_bound/ReadVariableOp2d
0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp0layer_1/gdn_1/gamma/lower_bound_1/ReadVariableOp2L
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
?
|
gdn_0_cond_2_true_199377'
#gdn_0_cond_2_identity_gdn_0_biasadd
gdn_0_cond_2_placeholder
gdn_0_cond_2_identity?
gdn_0/cond_2/IdentityIdentity#gdn_0_cond_2_identity_gdn_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2
gdn_0/cond_2/Identity"7
gdn_0_cond_2_identitygdn_0/cond_2/Identity:output:0*(
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
)analysis_layer_0_gdn_0_cond_1_true_198008C
?analysis_layer_0_gdn_0_cond_1_identity_analysis_layer_0_biasadd-
)analysis_layer_0_gdn_0_cond_1_placeholder*
&analysis_layer_0_gdn_0_cond_1_identity?
&analysis/layer_0/gdn_0/cond_1/IdentityIdentity?analysis_layer_0_gdn_0_cond_1_identity_analysis_layer_0_biasadd*
T0*B
_output_shapes0
.:,????????????????????????????2(
&analysis/layer_0/gdn_0/cond_1/Identity"Y
&analysis_layer_0_gdn_0_cond_1_identity/analysis/layer_0/gdn_0/cond_1/Identity:output:0*(
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
U
input_1J
serving_default_input_1:0+???????????????????????????W
analysisK
StatefulPartitionedCall:0,????????????????????????????tensorflow/serving/predict:??
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
_tf_keras_network??{"name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "name": "analysis", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["analysis", 1, 0]]}, "shared_object_id": 39, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "name": "analysis", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 38}], "input_layers": [["input_1", 0, 0]], "output_layers": [["analysis", 1, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
̈
layer-0
	layer_with_weights-0
	layer-1

layer_with_weights-1

layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"??
_tf_keras_sequential??{"name": "analysis", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, "float32", "lambda_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}, "shared_object_id": 1}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 2}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}, "shared_object_id": 3}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 13}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 14}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 23}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 24}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 33}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 34}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 37}]}}}
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
?
$regularization_losses
%trainable_variables
&	variables
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDEucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 2}
?
(_activation
)_kernel_parameter
_bias_parameter
*regularization_losses
+trainable_variables
,	variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}, "shared_object_id": 3}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
?
._activation
/_kernel_parameter
_bias_parameter
0regularization_losses
1trainable_variables
2	variables
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 14}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 23, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
4_activation
5_kernel_parameter
_bias_parameter
6regularization_losses
7trainable_variables
8	variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 24}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
:_kernel_parameter
_bias_parameter
;regularization_losses
<trainable_variables
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 34}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 37, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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
):'?2layer_0/gdn_0/reparam_beta
/:-
??2layer_0/gdn_0/reparam_gamma
&:$	?2layer_0/kernel_rdft
:?2layer_1/bias
):'?2layer_1/gdn_1/reparam_beta
/:-
??2layer_1/gdn_1/reparam_gamma
':%
??2layer_1/kernel_rdft
:?2layer_2/bias
):'?2layer_2/gdn_2/reparam_beta
/:-
??2layer_2/gdn_2/reparam_gamma
':%
??2layer_2/kernel_rdft
:?2layer_3/bias
':%
??2layer_3/kernel_rdft
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses
$regularization_losses
Enon_trainable_variables
Flayer_metrics

Glayers
%trainable_variables
Hmetrics
&	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
I_beta_parameter
J_gamma_parameter
Kregularization_losses
Ltrainable_variables
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "gdn_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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
Olayer_regularization_losses
*regularization_losses
Pnon_trainable_variables
Qlayer_metrics

Rlayers
+trainable_variables
Smetrics
,	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
T_beta_parameter
U_gamma_parameter
Vregularization_losses
Wtrainable_variables
X	variables
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "gdn_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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
Zlayer_regularization_losses
0regularization_losses
[non_trainable_variables
\layer_metrics

]layers
1trainable_variables
^metrics
2	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
__beta_parameter
`_gamma_parameter
aregularization_losses
btrainable_variables
c	variables
d	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "gdn_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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
elayer_regularization_losses
6regularization_losses
fnon_trainable_variables
glayer_metrics

hlayers
7trainable_variables
imetrics
8	variables
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
Kregularization_losses
pnon_trainable_variables
qlayer_metrics

rlayers
Ltrainable_variables
smetrics
M	variables
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
(0"
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
Vregularization_losses
unon_trainable_variables
vlayer_metrics

wlayers
Wtrainable_variables
xmetrics
X	variables
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
.0"
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
aregularization_losses
znon_trainable_variables
{layer_metrics

|layers
btrainable_variables
}metrics
c	variables
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
40"
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
!__inference__wrapped_model_196081?
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
annotations? *@?=
;?8
input_1+???????????????????????????
?2?
(__inference_encoder_layer_call_fn_197322
(__inference_encoder_layer_call_fn_197476?
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
C__inference_encoder_layer_call_and_return_conditional_losses_197090
C__inference_encoder_layer_call_and_return_conditional_losses_197167
C__inference_encoder_layer_call_and_return_conditional_losses_197979
C__inference_encoder_layer_call_and_return_conditional_losses_198403?
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
)__inference_analysis_layer_call_fn_196854
)__inference_analysis_layer_call_fn_197012?
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
D__inference_analysis_layer_call_and_return_conditional_losses_196608
D__inference_analysis_layer_call_and_return_conditional_losses_196695
D__inference_analysis_layer_call_and_return_conditional_losses_198827
D__inference_analysis_layer_call_and_return_conditional_losses_199251?
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
$__inference_signature_wrapper_197555input_1"?
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
B__inference_lambda_layer_call_and_return_conditional_losses_199257
B__inference_lambda_layer_call_and_return_conditional_losses_199263?
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
?2?
C__inference_layer_0_layer_call_and_return_conditional_losses_199407?
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
C__inference_layer_1_layer_call_and_return_conditional_losses_199551?
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
C__inference_layer_2_layer_call_and_return_conditional_losses_199695?
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
C__inference_layer_3_layer_call_and_return_conditional_losses_199713?
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

Const_21?
!__inference__wrapped_model_196081?:??????????????????????J?G
@?=
;?8
input_1+???????????????????????????
? "N?K
I
analysis=?:
analysis,?????????????????????????????
D__inference_analysis_layer_call_and_return_conditional_losses_196608?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
D__inference_analysis_layer_call_and_return_conditional_losses_196695?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
D__inference_analysis_layer_call_and_return_conditional_losses_198827?:??????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
D__inference_analysis_layer_call_and_return_conditional_losses_199251?:??????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_analysis_layer_call_fn_196854?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p

 
? "3?0,?????????????????????????????
)__inference_analysis_layer_call_fn_197012?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p 

 
? "3?0,?????????????????????????????
C__inference_encoder_layer_call_and_return_conditional_losses_197090?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_197167?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_197979?:??????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p

 
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_encoder_layer_call_and_return_conditional_losses_198403?:??????????????????????Q?N
G?D
:?7
inputs+???????????????????????????
p 

 
? "@?=
6?3
0,????????????????????????????
? ?
(__inference_encoder_layer_call_fn_197322?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p

 
? "3?0,?????????????????????????????
(__inference_encoder_layer_call_fn_197476?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "3?0,?????????????????????????????
B__inference_lambda_layer_call_and_return_conditional_losses_199257?Q?N
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
B__inference_lambda_layer_call_and_return_conditional_losses_199263?Q?N
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
C__inference_layer_0_layer_call_and_return_conditional_losses_199407????????I?F
??<
:?7
inputs+???????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_1_layer_call_and_return_conditional_losses_199551????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_2_layer_call_and_return_conditional_losses_199695????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_3_layer_call_and_return_conditional_losses_199713??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
$__inference_signature_wrapper_197555?:??????????????????????U?R
? 
K?H
F
input_1;?8
input_1+???????????????????????????"N?K
I
analysis=?:
analysis,????????????????????????????