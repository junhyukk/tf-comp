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
trainable_variables
regularization_losses
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
R
$trainable_variables
%regularization_losses
&	variables
'	keras_api
?
(_activation
)_kernel_parameter
_bias_parameter
*trainable_variables
+regularization_losses
,	variables
-	keras_api
?
._activation
/_kernel_parameter
_bias_parameter
0trainable_variables
1regularization_losses
2	variables
3	keras_api
?
4_activation
5_kernel_parameter
_bias_parameter
6trainable_variables
7regularization_losses
8	variables
9	keras_api
~
:_kernel_parameter
_bias_parameter
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

0
1
 
 
 
 
 
 
 
?

Dlayers
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics
$trainable_variables
%regularization_losses
&	variables
Hlayer_metrics
}
I_beta_parameter
J_gamma_parameter
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api


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

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
*trainable_variables
+regularization_losses
,	variables
Slayer_metrics
}
T_beta_parameter
U_gamma_parameter
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api


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

Zlayers
[layer_regularization_losses
\non_trainable_variables
]metrics
0trainable_variables
1regularization_losses
2	variables
^layer_metrics
}
__beta_parameter
`_gamma_parameter
atrainable_variables
bregularization_losses
c	variables
d	keras_api


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

elayers
flayer_regularization_losses
gnon_trainable_variables
hmetrics
6trainable_variables
7regularization_losses
8	variables
ilayer_metrics


rdft

0
1
 

0
1
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
 
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
Ktrainable_variables
Lregularization_losses
M	variables
slayer_metrics

(0
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
Vtrainable_variables
Wregularization_losses
X	variables
xlayer_metrics

.0
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
atrainable_variables
bregularization_losses
c	variables
}layer_metrics
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
$__inference_signature_wrapper_197594
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
__inference__traced_save_199869
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
"__inference__traced_restore_199921??"
?
W
gdn_1_cond_false_199467#
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
?
?
gdn_2_cond_1_cond_true_199630!
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
?
?
gdn_1_cond_1_cond_false_196333"
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
#gdn_1_cond_1_cond_cond_false_196343*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_1_cond_1_cond_cond_true_1963422
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
?
?
/analysis_layer_0_gdn_0_cond_2_cond_false_197716I
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
?
?
%layer_2_gdn_2_cond_1_cond_true_1991761
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
?
?
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_197778K
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
?	
?
7encoder_analysis_layer_0_gdn_0_cond_2_cond_false_195818Y
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
?
?
!layer_2_gdn_2_cond_2_false_1992513
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
&layer_2_gdn_2_cond_2_cond_false_199260*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_2_cond_true_1992592
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
?
?
1encoder_analysis_layer_0_gdn_0_cond_2_true_195808Y
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
?
?
)analysis_layer_1_gdn_1_cond_1_true_198183C
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
?
?
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_197642K
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
?
|
gdn_2_cond_2_true_199704'
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
?
?
1encoder_analysis_layer_1_gdn_1_cond_2_true_195944Y
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
?
?
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_198339H
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
?
v
gdn_0_cond_1_true_196159!
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
?
?
"gdn_2_cond_1_cond_cond_true_199640)
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
?
?
/analysis_layer_1_gdn_1_cond_2_cond_false_198276I
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
?
?
)analysis_layer_2_gdn_2_cond_1_true_198319C
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
?
?
7encoder_analysis_layer_1_gdn_1_cond_1_cond_false_195871T
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
<encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_false_195881*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_true_19588021
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
?N
?
C__inference_layer_0_layer_call_and_return_conditional_losses_199446

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
gdn_0_cond_false_199323*
output_shapes
: *)
then_branchR
gdn_0_cond_true_1993222

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
gdn_0_cond_1_false_199334*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_1_true_1993332
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
_gradient_op_typeCustomGradient-199379*.
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
_gradient_op_typeCustomGradient-199389*.
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
_gradient_op_typeCustomGradient-199403*$
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
gdn_0_cond_2_false_199417*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_2_true_1994162
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
?
?
*analysis_layer_1_gdn_1_cond_2_false_198267E
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
/analysis_layer_1_gdn_1_cond_2_cond_false_198276*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_2_cond_true_1982752$
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
?
?
/analysis_layer_0_gdn_0_cond_1_cond_false_198057D
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
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_198067*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_1980662)
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
?
?
*analysis_layer_2_gdn_2_cond_2_false_198403E
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
/analysis_layer_2_gdn_2_cond_2_cond_false_198412*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_2_cond_true_1984112$
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
 layer_0_gdn_0_cond_1_true_1988951
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
gdn_2_cond_2_cond_true_199713(
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
?
?
gdn_0_cond_1_cond_false_199343"
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
#gdn_0_cond_1_cond_cond_false_199353*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_0_cond_1_cond_cond_true_1993522
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
?
?
/analysis_layer_2_gdn_2_cond_2_cond_false_197988I
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
?(
?
__inference__traced_save_199869
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
?
?
%layer_1_gdn_1_cond_2_cond_true_1991238
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
?
?
!layer_2_gdn_2_cond_1_false_198744-
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
&layer_2_gdn_2_cond_1_cond_false_198753*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_1_cond_true_1987522
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
?
u
gdn_1_cond_2_false_196407#
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
gdn_1_cond_2_cond_false_196416*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_2_cond_true_1964152
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
?
?
*analysis_layer_0_gdn_0_cond_2_false_198131E
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
/analysis_layer_0_gdn_0_cond_2_cond_false_198140*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_2_cond_true_1981392$
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
"gdn_1_cond_1_cond_cond_true_199496)
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
?
?
 layer_1_gdn_1_cond_1_true_1990311
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
?
?
gdn_0_cond_2_cond_true_196251(
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
?
?
gdn_0_cond_2_cond_false_199426'
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
?
?
#gdn_1_cond_1_cond_cond_false_199497&
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
?
?
gdn_2_cond_2_cond_true_196579(
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
?
?
*analysis_layer_2_gdn_2_cond_1_false_198320?
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
/analysis_layer_2_gdn_2_cond_1_cond_false_198329*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_1_cond_true_1983282$
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
?
?
 layer_1_gdn_1_cond_2_true_1986907
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
?
?
/analysis_layer_0_gdn_0_cond_1_cond_false_197633D
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
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_197643*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_1976422)
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
%layer_1_gdn_1_cond_1_cond_true_1990401
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
?
?
"gdn_2_cond_1_cond_cond_true_196506)
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
?
?
)analysis_layer_1_gdn_1_cond_2_true_198266I
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
o
gdn_2_cond_1_false_196488
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
gdn_2_cond_1_cond_false_196497*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_1_cond_true_1964962
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
e
layer_0_gdn_0_cond_true_198884"
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
?
?
2encoder_analysis_layer_0_gdn_0_cond_2_false_195809U
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
7encoder_analysis_layer_0_gdn_0_cond_2_cond_false_195818*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_0_gdn_0_cond_2_cond_true_1958172,
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
?
?
1encoder_analysis_layer_0_gdn_0_cond_1_true_195725S
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
?
?
/analysis_layer_2_gdn_2_cond_1_cond_false_197905D
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
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_197915*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_1979142)
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
;encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_true_196016[
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
?N
?
C__inference_layer_0_layer_call_and_return_conditional_losses_196272

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
gdn_0_cond_false_196149*
output_shapes
: *)
then_branchR
gdn_0_cond_true_1961482

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
gdn_0_cond_1_false_196160*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_1_true_1961592
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
_gradient_op_typeCustomGradient-196205*.
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
_gradient_op_typeCustomGradient-196215*.
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
_gradient_op_typeCustomGradient-196229*$
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
gdn_0_cond_2_false_196243*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_0_cond_2_true_1962422
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
?
w
layer_0_gdn_0_cond_false_1984613
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
?
?
 layer_0_gdn_0_cond_2_true_1989787
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
?
?
%layer_0_gdn_0_cond_2_cond_true_1985638
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
?
?
'analysis_layer_2_gdn_2_cond_true_197884+
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
?
(analysis_layer_1_gdn_1_cond_false_198173E
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
?
?
 layer_2_gdn_2_cond_2_true_1992507
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
?
W
gdn_1_cond_false_196313#
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
?
?
2encoder_analysis_layer_1_gdn_1_cond_2_false_195945U
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
7encoder_analysis_layer_1_gdn_1_cond_2_cond_false_195954*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_1_gdn_1_cond_2_cond_true_1959532,
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
?
?
&layer_2_gdn_2_cond_2_cond_false_1992607
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
*layer_2_gdn_2_cond_1_cond_cond_true_1987629
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
?
0encoder_analysis_layer_2_gdn_2_cond_false_195987U
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
?
?
(analysis_layer_2_gdn_2_cond_false_197885E
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
?
v
gdn_1_cond_1_true_199477!
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
?
?
 layer_2_gdn_2_cond_1_true_1991671
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
?
?
gdn_1_cond_1_cond_false_199487"
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
#gdn_1_cond_1_cond_cond_false_199497*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_1_cond_1_cond_cond_true_1994962
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
?
M
gdn_1_cond_true_196312
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
?
?
)__inference_analysis_layer_call_fn_197051
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
D__inference_analysis_layer_call_and_return_conditional_losses_1969762
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
?
?
 layer_2_gdn_2_cond_1_true_1987431
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
?
?
 layer_1_gdn_1_cond_1_true_1986071
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
?
C__inference_layer_3_layer_call_and_return_conditional_losses_199752

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
?
?
.analysis_layer_2_gdn_2_cond_2_cond_true_197987J
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
(analysis_layer_1_gdn_1_cond_false_197749E
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
?
v
gdn_1_cond_1_true_196323!
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
?
e
layer_1_gdn_1_cond_true_198596"
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
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197129
input_1
analysis_197055"
analysis_197057:	?
analysis_197059:	?
analysis_197061#
analysis_197063:
??
analysis_197065
analysis_197067
analysis_197069:	?
analysis_197071
analysis_197073
analysis_197075
analysis_197077#
analysis_197079:
??
analysis_197081:	?
analysis_197083#
analysis_197085:
??
analysis_197087
analysis_197089
analysis_197091:	?
analysis_197093
analysis_197095
analysis_197097
analysis_197099#
analysis_197101:
??
analysis_197103:	?
analysis_197105#
analysis_197107:
??
analysis_197109
analysis_197111
analysis_197113:	?
analysis_197115
analysis_197117
analysis_197119
analysis_197121#
analysis_197123:
??
analysis_197125:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinput_1analysis_197055analysis_197057analysis_197059analysis_197061analysis_197063analysis_197065analysis_197067analysis_197069analysis_197071analysis_197073analysis_197075analysis_197077analysis_197079analysis_197081analysis_197083analysis_197085analysis_197087analysis_197089analysis_197091analysis_197093analysis_197095analysis_197097analysis_197099analysis_197101analysis_197103analysis_197105analysis_197107analysis_197109analysis_197111analysis_197113analysis_197115analysis_197117analysis_197119analysis_197121analysis_197123analysis_197125*0
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
D__inference_analysis_layer_call_and_return_conditional_losses_1968182"
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
?
o
gdn_1_cond_1_false_199478
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
gdn_1_cond_1_cond_false_199487*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_1_cond_true_1994862
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
?
?
6encoder_analysis_layer_2_gdn_2_cond_1_cond_true_196006S
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
?
?
gdn_1_cond_2_cond_false_199570'
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
?
?
!layer_1_gdn_1_cond_2_false_1986913
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
&layer_1_gdn_1_cond_2_cond_false_198700*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_2_cond_true_1986992
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
?
e
layer_2_gdn_2_cond_true_198732"
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
B__inference_lambda_layer_call_and_return_conditional_losses_196655

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
?
?
&layer_1_gdn_1_cond_2_cond_false_1991247
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
?
?
gdn_1_cond_1_cond_true_199486!
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
?
?
+layer_0_gdn_0_cond_1_cond_cond_false_1989156
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
?
?
$__inference_signature_wrapper_197594
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
!__inference__wrapped_model_1961202
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
?
M
gdn_2_cond_true_196476
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
?
?
2encoder_analysis_layer_0_gdn_0_cond_1_false_195726O
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
7encoder_analysis_layer_0_gdn_0_cond_1_cond_false_195735*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_0_gdn_0_cond_1_cond_true_1957342,
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
?
?
&layer_0_gdn_0_cond_2_cond_false_1985647
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
?
?
/encoder_analysis_layer_2_gdn_2_cond_true_1959863
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
?
?
6encoder_analysis_layer_0_gdn_0_cond_2_cond_true_195817Z
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
?
?
)analysis_layer_2_gdn_2_cond_1_true_197895C
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
u
gdn_2_cond_2_false_199705#
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
gdn_2_cond_2_cond_false_199714*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_2_cond_true_1997132
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
?
?
+layer_2_gdn_2_cond_1_cond_cond_false_1987636
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
?
?
 layer_0_gdn_0_cond_1_true_1984711
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
?
?
!layer_1_gdn_1_cond_1_false_198608-
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
&layer_1_gdn_1_cond_1_cond_false_198617*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_1_cond_true_1986162
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
.analysis_layer_0_gdn_0_cond_1_cond_true_198056C
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
#gdn_0_cond_1_cond_cond_false_196179&
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
?
?
 layer_0_gdn_0_cond_2_true_1985547
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
?
w
layer_1_gdn_1_cond_false_1985973
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
?
?
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_198203H
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
?
?
/analysis_layer_2_gdn_2_cond_1_cond_false_198329D
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
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_198339*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_1983382)
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
??
?
D__inference_analysis_layer_call_and_return_conditional_losses_199290

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
layer_0_gdn_0_cond_false_198885*
output_shapes
: *1
then_branch"R 
layer_0_gdn_0_cond_true_1988842
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
!layer_0_gdn_0_cond_1_false_198896*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_1_true_1988952
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
_gradient_op_typeCustomGradient-198941*.
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
_gradient_op_typeCustomGradient-198951*.
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
_gradient_op_typeCustomGradient-198965*$
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
!layer_0_gdn_0_cond_2_false_198979*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_2_true_1989782
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
layer_1_gdn_1_cond_false_199021*
output_shapes
: *1
then_branch"R 
layer_1_gdn_1_cond_true_1990202
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
!layer_1_gdn_1_cond_1_false_199032*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_1_true_1990312
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
_gradient_op_typeCustomGradient-199077*.
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
_gradient_op_typeCustomGradient-199087*.
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
_gradient_op_typeCustomGradient-199101*$
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
!layer_1_gdn_1_cond_2_false_199115*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_2_true_1991142
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
layer_2_gdn_2_cond_false_199157*
output_shapes
: *1
then_branch"R 
layer_2_gdn_2_cond_true_1991562
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
!layer_2_gdn_2_cond_1_false_199168*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_1_true_1991672
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
_gradient_op_typeCustomGradient-199213*.
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
_gradient_op_typeCustomGradient-199223*.
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
_gradient_op_typeCustomGradient-199237*$
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
!layer_2_gdn_2_cond_2_false_199251*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_2_true_1992502
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
?
?
!layer_2_gdn_2_cond_2_false_1988273
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
&layer_2_gdn_2_cond_2_cond_false_198836*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_2_cond_true_1988352
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
?
W
gdn_0_cond_false_196149#
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
?
?
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_197643H
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
o
gdn_1_cond_1_false_196324
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
gdn_1_cond_1_cond_false_196333*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_1_cond_true_1963322
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
?
?
gdn_1_cond_1_cond_true_196332!
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
?*
?
D__inference_analysis_layer_call_and_return_conditional_losses_196818

inputs
layer_0_196741!
layer_0_196743:	?
layer_0_196745:	?
layer_0_196747"
layer_0_196749:
??
layer_0_196751
layer_0_196753
layer_0_196755:	?
layer_0_196757
layer_0_196759
layer_0_196761
layer_1_196764"
layer_1_196766:
??
layer_1_196768:	?
layer_1_196770"
layer_1_196772:
??
layer_1_196774
layer_1_196776
layer_1_196778:	?
layer_1_196780
layer_1_196782
layer_1_196784
layer_2_196787"
layer_2_196789:
??
layer_2_196791:	?
layer_2_196793"
layer_2_196795:
??
layer_2_196797
layer_2_196799
layer_2_196801:	?
layer_2_196803
layer_2_196805
layer_2_196807
layer_3_196810"
layer_3_196812:
??
layer_3_196814:	?
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
B__inference_lambda_layer_call_and_return_conditional_losses_1961302
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196741layer_0_196743layer_0_196745layer_0_196747layer_0_196749layer_0_196751layer_0_196753layer_0_196755layer_0_196757layer_0_196759layer_0_196761*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962722!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196764layer_1_196766layer_1_196768layer_1_196770layer_1_196772layer_1_196774layer_1_196776layer_1_196778layer_1_196780layer_1_196782layer_1_196784*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1964362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196787layer_2_196789layer_2_196791layer_2_196793layer_2_196795layer_2_196797layer_2_196799layer_2_196801layer_2_196803layer_2_196805layer_2_196807*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1966002!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196810layer_3_196812layer_3_196814*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_1966382!
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
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_198202K
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
/encoder_analysis_layer_0_gdn_0_cond_true_1957143
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
?
?
#gdn_2_cond_1_cond_cond_false_196507&
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
?*
?
D__inference_analysis_layer_call_and_return_conditional_losses_196976

inputs
layer_0_196899!
layer_0_196901:	?
layer_0_196903:	?
layer_0_196905"
layer_0_196907:
??
layer_0_196909
layer_0_196911
layer_0_196913:	?
layer_0_196915
layer_0_196917
layer_0_196919
layer_1_196922"
layer_1_196924:
??
layer_1_196926:	?
layer_1_196928"
layer_1_196930:
??
layer_1_196932
layer_1_196934
layer_1_196936:	?
layer_1_196938
layer_1_196940
layer_1_196942
layer_2_196945"
layer_2_196947:
??
layer_2_196949:	?
layer_2_196951"
layer_2_196953:
??
layer_2_196955
layer_2_196957
layer_2_196959:	?
layer_2_196961
layer_2_196963
layer_2_196965
layer_3_196968"
layer_3_196970:
??
layer_3_196972:	?
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
B__inference_lambda_layer_call_and_return_conditional_losses_1966552
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196899layer_0_196901layer_0_196903layer_0_196905layer_0_196907layer_0_196909layer_0_196911layer_0_196913layer_0_196915layer_0_196917layer_0_196919*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962722!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196922layer_1_196924layer_1_196926layer_1_196928layer_1_196930layer_1_196932layer_1_196934layer_1_196936layer_1_196938layer_1_196940layer_1_196942*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1964362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196945layer_2_196947layer_2_196949layer_2_196951layer_2_196953layer_2_196955layer_2_196957layer_2_196959layer_2_196961layer_2_196963layer_2_196965*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1966002!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196968layer_3_196970layer_3_196972*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_1966382!
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
?
u
gdn_0_cond_2_false_196243#
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
gdn_0_cond_2_cond_false_196252*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_2_cond_true_1962512
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
?
?
.analysis_layer_1_gdn_1_cond_1_cond_true_198192C
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
?
?
*analysis_layer_0_gdn_0_cond_1_false_198048?
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
/analysis_layer_0_gdn_0_cond_1_cond_false_198057*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_1_cond_true_1980562$
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
o
gdn_0_cond_1_false_196160
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
gdn_0_cond_1_cond_false_196169*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_1_cond_true_1961682
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
?
?
gdn_2_cond_2_cond_false_199714'
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
gdn_0_cond_1_cond_true_199342!
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
?
?
%layer_1_gdn_1_cond_2_cond_true_1986998
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
?
?
/analysis_layer_1_gdn_1_cond_1_cond_false_197769D
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
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_197779*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_1977782)
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
)analysis_layer_0_gdn_0_cond_1_true_197623C
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
?
?
&layer_2_gdn_2_cond_1_cond_false_1991772
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
+layer_2_gdn_2_cond_1_cond_cond_false_199187*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_2_gdn_2_cond_1_cond_cond_true_1991862 
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
?
?
+layer_2_gdn_2_cond_1_cond_cond_false_1991876
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
?
?
6encoder_analysis_layer_2_gdn_2_cond_2_cond_true_196089Z
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
?
?
.analysis_layer_2_gdn_2_cond_2_cond_true_198411J
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
e
layer_0_gdn_0_cond_true_198460"
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
|
gdn_0_cond_2_true_196242'
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
?
?
(analysis_layer_2_gdn_2_cond_false_198309E
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
?
?
+layer_1_gdn_1_cond_1_cond_cond_false_1990516
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
?N
?
C__inference_layer_1_layer_call_and_return_conditional_losses_199590

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
gdn_1_cond_false_199467*
output_shapes
: *)
then_branchR
gdn_1_cond_true_1994662

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
gdn_1_cond_1_false_199478*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_1_true_1994772
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
_gradient_op_typeCustomGradient-199523*.
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
_gradient_op_typeCustomGradient-199533*.
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
_gradient_op_typeCustomGradient-199547*$
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
gdn_1_cond_2_false_199561*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_2_true_1995602
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
e
layer_1_gdn_1_cond_true_199020"
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
?
?
)analysis_layer_1_gdn_1_cond_2_true_197842I
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
?
?
2encoder_analysis_layer_2_gdn_2_cond_1_false_195998O
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
7encoder_analysis_layer_2_gdn_2_cond_1_cond_false_196007*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_2_gdn_2_cond_1_cond_true_1960062,
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
?
?
*layer_0_gdn_0_cond_1_cond_cond_true_1984909
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
?
?
!layer_0_gdn_0_cond_1_false_198896-
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
&layer_0_gdn_0_cond_1_cond_false_198905*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_1_cond_true_1989042
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
??
?
D__inference_analysis_layer_call_and_return_conditional_losses_198866

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
layer_0_gdn_0_cond_false_198461*
output_shapes
: *1
then_branch"R 
layer_0_gdn_0_cond_true_1984602
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
!layer_0_gdn_0_cond_1_false_198472*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_1_true_1984712
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
_gradient_op_typeCustomGradient-198517*.
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
_gradient_op_typeCustomGradient-198527*.
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
_gradient_op_typeCustomGradient-198541*$
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
!layer_0_gdn_0_cond_2_false_198555*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_0_gdn_0_cond_2_true_1985542
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
layer_1_gdn_1_cond_false_198597*
output_shapes
: *1
then_branch"R 
layer_1_gdn_1_cond_true_1985962
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
!layer_1_gdn_1_cond_1_false_198608*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_1_true_1986072
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
_gradient_op_typeCustomGradient-198653*.
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
_gradient_op_typeCustomGradient-198663*.
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
_gradient_op_typeCustomGradient-198677*$
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
!layer_1_gdn_1_cond_2_false_198691*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_1_gdn_1_cond_2_true_1986902
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
layer_2_gdn_2_cond_false_198733*
output_shapes
: *1
then_branch"R 
layer_2_gdn_2_cond_true_1987322
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
!layer_2_gdn_2_cond_1_false_198744*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_1_true_1987432
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
_gradient_op_typeCustomGradient-198789*.
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
_gradient_op_typeCustomGradient-198799*.
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
_gradient_op_typeCustomGradient-198813*$
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
!layer_2_gdn_2_cond_2_false_198827*A
output_shapes0
.:,????????????????????????????*3
then_branch$R"
 layer_2_gdn_2_cond_2_true_1988262
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
)analysis_layer_2_gdn_2_cond_2_true_197978I
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
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_198442

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
(analysis_layer_0_gdn_0_cond_false_198037*
output_shapes
: *:
then_branch+R)
'analysis_layer_0_gdn_0_cond_true_1980362
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
*analysis_layer_0_gdn_0_cond_1_false_198048*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_1_true_1980472
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
_gradient_op_typeCustomGradient-198093*.
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
_gradient_op_typeCustomGradient-198103*.
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
_gradient_op_typeCustomGradient-198117*$
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
*analysis_layer_0_gdn_0_cond_2_false_198131*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_2_true_1981302
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
(analysis_layer_1_gdn_1_cond_false_198173*
output_shapes
: *:
then_branch+R)
'analysis_layer_1_gdn_1_cond_true_1981722
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
*analysis_layer_1_gdn_1_cond_1_false_198184*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_1_true_1981832
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
_gradient_op_typeCustomGradient-198229*.
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
_gradient_op_typeCustomGradient-198239*.
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
_gradient_op_typeCustomGradient-198253*$
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
*analysis_layer_1_gdn_1_cond_2_false_198267*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_2_true_1982662
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
(analysis_layer_2_gdn_2_cond_false_198309*
output_shapes
: *:
then_branch+R)
'analysis_layer_2_gdn_2_cond_true_1983082
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
*analysis_layer_2_gdn_2_cond_1_false_198320*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_1_true_1983192
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
_gradient_op_typeCustomGradient-198365*.
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
_gradient_op_typeCustomGradient-198375*.
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
_gradient_op_typeCustomGradient-198389*$
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
*analysis_layer_2_gdn_2_cond_2_false_198403*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_2_true_1984022
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
?
o
gdn_0_cond_1_false_199334
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
gdn_0_cond_1_cond_false_199343*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_1_cond_true_1993422
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
?
?
.analysis_layer_0_gdn_0_cond_1_cond_true_197632C
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
#gdn_0_cond_1_cond_cond_false_199353&
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
?
?
!layer_1_gdn_1_cond_2_false_1991153
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
&layer_1_gdn_1_cond_2_cond_false_199124*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_2_cond_true_1991232
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
?
u
gdn_0_cond_2_false_199417#
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
gdn_0_cond_2_cond_false_199426*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_0_cond_2_cond_true_1994252
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
?
|
gdn_1_cond_2_true_199560'
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
?
?
!layer_2_gdn_2_cond_1_false_199168-
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
&layer_2_gdn_2_cond_1_cond_false_199177*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_2_gdn_2_cond_1_cond_true_1991762
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
?
?
2encoder_analysis_layer_1_gdn_1_cond_1_false_195862O
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
7encoder_analysis_layer_1_gdn_1_cond_1_cond_false_195871*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_1_gdn_1_cond_1_cond_true_1958702,
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
?
?
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_198338K
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
?
M
gdn_0_cond_true_196148
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
?	
?
<encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_false_195881X
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
<encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_false_195745X
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
?
?
*layer_2_gdn_2_cond_1_cond_cond_true_1991869
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
?N
?
C__inference_layer_2_layer_call_and_return_conditional_losses_199734

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
gdn_2_cond_false_199611*
output_shapes
: *)
then_branchR
gdn_2_cond_true_1996102

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
gdn_2_cond_1_false_199622*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_1_true_1996212
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
_gradient_op_typeCustomGradient-199667*.
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
_gradient_op_typeCustomGradient-199677*.
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
_gradient_op_typeCustomGradient-199691*$
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
gdn_2_cond_2_false_199705*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_2_true_1997042
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
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197206
input_1
analysis_197132"
analysis_197134:	?
analysis_197136:	?
analysis_197138#
analysis_197140:
??
analysis_197142
analysis_197144
analysis_197146:	?
analysis_197148
analysis_197150
analysis_197152
analysis_197154#
analysis_197156:
??
analysis_197158:	?
analysis_197160#
analysis_197162:
??
analysis_197164
analysis_197166
analysis_197168:	?
analysis_197170
analysis_197172
analysis_197174
analysis_197176#
analysis_197178:
??
analysis_197180:	?
analysis_197182#
analysis_197184:
??
analysis_197186
analysis_197188
analysis_197190:	?
analysis_197192
analysis_197194
analysis_197196
analysis_197198#
analysis_197200:
??
analysis_197202:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinput_1analysis_197132analysis_197134analysis_197136analysis_197138analysis_197140analysis_197142analysis_197144analysis_197146analysis_197148analysis_197150analysis_197152analysis_197154analysis_197156analysis_197158analysis_197160analysis_197162analysis_197164analysis_197166analysis_197168analysis_197170analysis_197172analysis_197174analysis_197176analysis_197178analysis_197180analysis_197182analysis_197184analysis_197186analysis_197188analysis_197190analysis_197192analysis_197194analysis_197196analysis_197198analysis_197200analysis_197202*0
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
D__inference_analysis_layer_call_and_return_conditional_losses_1969762"
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
^
B__inference_lambda_layer_call_and_return_conditional_losses_196130

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
?
?
&layer_1_gdn_1_cond_2_cond_false_1987007
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
?
?
(analysis_layer_0_gdn_0_cond_false_197613E
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
?
?
*analysis_layer_0_gdn_0_cond_2_false_197707E
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
/analysis_layer_0_gdn_0_cond_2_cond_false_197716*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_2_cond_true_1977152$
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
?
?
&layer_2_gdn_2_cond_2_cond_false_1988367
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
?	
?
7encoder_analysis_layer_1_gdn_1_cond_2_cond_false_195954Y
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
?
?
.analysis_layer_1_gdn_1_cond_2_cond_true_198275J
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
?
?
gdn_0_cond_2_cond_false_196252'
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
e
layer_2_gdn_2_cond_true_199156"
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
?
?
%layer_2_gdn_2_cond_2_cond_true_1988358
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
?
?
*analysis_layer_1_gdn_1_cond_1_false_198184?
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
/analysis_layer_1_gdn_1_cond_1_cond_false_198193*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_1_cond_true_1981922$
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
?
?
6encoder_analysis_layer_0_gdn_0_cond_1_cond_true_195734S
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
?
?
*layer_0_gdn_0_cond_1_cond_cond_true_1989149
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
?
?
/analysis_layer_2_gdn_2_cond_2_cond_false_198412I
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
?*
?
D__inference_analysis_layer_call_and_return_conditional_losses_196647
lambda_input
layer_0_196273!
layer_0_196275:	?
layer_0_196277:	?
layer_0_196279"
layer_0_196281:
??
layer_0_196283
layer_0_196285
layer_0_196287:	?
layer_0_196289
layer_0_196291
layer_0_196293
layer_1_196437"
layer_1_196439:
??
layer_1_196441:	?
layer_1_196443"
layer_1_196445:
??
layer_1_196447
layer_1_196449
layer_1_196451:	?
layer_1_196453
layer_1_196455
layer_1_196457
layer_2_196601"
layer_2_196603:
??
layer_2_196605:	?
layer_2_196607"
layer_2_196609:
??
layer_2_196611
layer_2_196613
layer_2_196615:	?
layer_2_196617
layer_2_196619
layer_2_196621
layer_3_196639"
layer_3_196641:
??
layer_3_196643:	?
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
B__inference_lambda_layer_call_and_return_conditional_losses_1961302
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196273layer_0_196275layer_0_196277layer_0_196279layer_0_196281layer_0_196283layer_0_196285layer_0_196287layer_0_196289layer_0_196291layer_0_196293*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962722!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196437layer_1_196439layer_1_196441layer_1_196443layer_1_196445layer_1_196447layer_1_196449layer_1_196451layer_1_196453layer_1_196455layer_1_196457*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1964362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196601layer_2_196603layer_2_196605layer_2_196607layer_2_196609layer_2_196611layer_2_196613layer_2_196615layer_2_196617layer_2_196619layer_2_196621*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1966002!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196639layer_3_196641layer_3_196643*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_1966382!
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
^
B__inference_lambda_layer_call_and_return_conditional_losses_199302

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
?
?
C__inference_layer_3_layer_call_and_return_conditional_losses_196638

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
?
?
gdn_1_cond_2_cond_true_199569(
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
gdn_0_cond_1_cond_true_196168!
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
?
?
1encoder_analysis_layer_1_gdn_1_cond_1_true_195861S
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
?
?
gdn_2_cond_1_cond_true_196496!
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
?
?
4analysis_layer_0_gdn_0_cond_1_cond_cond_false_198067H
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
?
6encoder_analysis_layer_1_gdn_1_cond_2_cond_true_195953Z
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
?*
?
D__inference_analysis_layer_call_and_return_conditional_losses_196734
lambda_input
layer_0_196657!
layer_0_196659:	?
layer_0_196661:	?
layer_0_196663"
layer_0_196665:
??
layer_0_196667
layer_0_196669
layer_0_196671:	?
layer_0_196673
layer_0_196675
layer_0_196677
layer_1_196680"
layer_1_196682:
??
layer_1_196684:	?
layer_1_196686"
layer_1_196688:
??
layer_1_196690
layer_1_196692
layer_1_196694:	?
layer_1_196696
layer_1_196698
layer_1_196700
layer_2_196703"
layer_2_196705:
??
layer_2_196707:	?
layer_2_196709"
layer_2_196711:
??
layer_2_196713
layer_2_196715
layer_2_196717:	?
layer_2_196719
layer_2_196721
layer_2_196723
layer_3_196726"
layer_3_196728:
??
layer_3_196730:	?
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
B__inference_lambda_layer_call_and_return_conditional_losses_1966552
lambda/PartitionedCall?
layer_0/StatefulPartitionedCallStatefulPartitionedCalllambda/PartitionedCall:output:0layer_0_196657layer_0_196659layer_0_196661layer_0_196663layer_0_196665layer_0_196667layer_0_196669layer_0_196671layer_0_196673layer_0_196675layer_0_196677*
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
C__inference_layer_0_layer_call_and_return_conditional_losses_1962722!
layer_0/StatefulPartitionedCall?
layer_1/StatefulPartitionedCallStatefulPartitionedCall(layer_0/StatefulPartitionedCall:output:0layer_1_196680layer_1_196682layer_1_196684layer_1_196686layer_1_196688layer_1_196690layer_1_196692layer_1_196694layer_1_196696layer_1_196698layer_1_196700*
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
C__inference_layer_1_layer_call_and_return_conditional_losses_1964362!
layer_1/StatefulPartitionedCall?
layer_2/StatefulPartitionedCallStatefulPartitionedCall(layer_1/StatefulPartitionedCall:output:0layer_2_196703layer_2_196705layer_2_196707layer_2_196709layer_2_196711layer_2_196713layer_2_196715layer_2_196717layer_2_196719layer_2_196721layer_2_196723*
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
C__inference_layer_2_layer_call_and_return_conditional_losses_1966002!
layer_2/StatefulPartitionedCall?
layer_3/StatefulPartitionedCallStatefulPartitionedCall(layer_2/StatefulPartitionedCall:output:0layer_3_196726layer_3_196728layer_3_196730*
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
C__inference_layer_3_layer_call_and_return_conditional_losses_1966382!
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
)analysis_layer_0_gdn_0_cond_1_true_198047C
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
?	
?
;encoder_analysis_layer_1_gdn_1_cond_1_cond_cond_true_195880[
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
?
?
gdn_1_cond_2_cond_true_196415(
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
?
?
&layer_0_gdn_0_cond_1_cond_false_1984812
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
+layer_0_gdn_0_cond_1_cond_cond_false_198491*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_0_gdn_0_cond_1_cond_cond_true_1984902 
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
?
?
*analysis_layer_1_gdn_1_cond_1_false_197760?
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
/analysis_layer_1_gdn_1_cond_1_cond_false_197769*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_1_cond_true_1977682$
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
?
?
&layer_1_gdn_1_cond_1_cond_false_1990412
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
+layer_1_gdn_1_cond_1_cond_cond_false_199051*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_1_gdn_1_cond_1_cond_cond_true_1990502 
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
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197440

inputs
analysis_197366"
analysis_197368:	?
analysis_197370:	?
analysis_197372#
analysis_197374:
??
analysis_197376
analysis_197378
analysis_197380:	?
analysis_197382
analysis_197384
analysis_197386
analysis_197388#
analysis_197390:
??
analysis_197392:	?
analysis_197394#
analysis_197396:
??
analysis_197398
analysis_197400
analysis_197402:	?
analysis_197404
analysis_197406
analysis_197408
analysis_197410#
analysis_197412:
??
analysis_197414:	?
analysis_197416#
analysis_197418:
??
analysis_197420
analysis_197422
analysis_197424:	?
analysis_197426
analysis_197428
analysis_197430
analysis_197432#
analysis_197434:
??
analysis_197436:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinputsanalysis_197366analysis_197368analysis_197370analysis_197372analysis_197374analysis_197376analysis_197378analysis_197380analysis_197382analysis_197384analysis_197386analysis_197388analysis_197390analysis_197392analysis_197394analysis_197396analysis_197398analysis_197400analysis_197402analysis_197404analysis_197406analysis_197408analysis_197410analysis_197412analysis_197414analysis_197416analysis_197418analysis_197420analysis_197422analysis_197424analysis_197426analysis_197428analysis_197430analysis_197432analysis_197434analysis_197436*0
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
D__inference_analysis_layer_call_and_return_conditional_losses_1969762"
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
?
u
gdn_1_cond_2_false_199561#
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
gdn_1_cond_2_cond_false_199570*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_1_cond_2_cond_true_1995692
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
?
?
(__inference_encoder_layer_call_fn_197361
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
C__inference_encoder_layer_call_and_return_conditional_losses_1972862
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
?
?
1encoder_analysis_layer_2_gdn_2_cond_2_true_196080Y
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
?
?
)analysis_layer_1_gdn_1_cond_1_true_197759C
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
?
M
gdn_2_cond_true_199610
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
?
u
gdn_2_cond_2_false_196571#
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
gdn_2_cond_2_cond_false_196580*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_2_cond_true_1965792
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
?
?
/analysis_layer_1_gdn_1_cond_2_cond_false_197852I
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
?	
?
7encoder_analysis_layer_2_gdn_2_cond_2_cond_false_196090Y
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
%layer_2_gdn_2_cond_1_cond_true_1987521
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
?
?
*analysis_layer_2_gdn_2_cond_1_false_197896?
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
/analysis_layer_2_gdn_2_cond_1_cond_false_197905*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_1_cond_true_1979042$
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
?
?
)analysis_layer_0_gdn_0_cond_2_true_198130I
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
?
?
'analysis_layer_1_gdn_1_cond_true_198172+
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
?
 layer_2_gdn_2_cond_2_true_1988267
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
?
W
gdn_0_cond_false_199323#
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
?
?
*layer_1_gdn_1_cond_1_cond_cond_true_1990509
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
"gdn_1_cond_1_cond_cond_true_196342)
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
?
?
.analysis_layer_1_gdn_1_cond_1_cond_true_197768C
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
?
o
gdn_2_cond_1_false_199622
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
gdn_2_cond_1_cond_false_199631*A
output_shapes0
.:,????????????????????????????*0
then_branch!R
gdn_2_cond_1_cond_true_1996302
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
gdn_2_cond_1_true_199621!
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
?
^
B__inference_lambda_layer_call_and_return_conditional_losses_199296

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
?
?
7encoder_analysis_layer_2_gdn_2_cond_1_cond_false_196007T
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
<encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_false_196017*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_true_19601621
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
?
?
!layer_0_gdn_0_cond_2_false_1989793
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
&layer_0_gdn_0_cond_2_cond_false_198988*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_2_cond_true_1989872
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
?
?
*analysis_layer_1_gdn_1_cond_2_false_197843E
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
/analysis_layer_1_gdn_1_cond_2_cond_false_197852*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_1_gdn_1_cond_2_cond_true_1978512$
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
?
?
gdn_0_cond_1_cond_false_196169"
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
#gdn_0_cond_1_cond_cond_false_196179*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_0_cond_1_cond_cond_true_1961782
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
?	
?
;encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_true_195744[
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
?
?
*analysis_layer_2_gdn_2_cond_2_false_197979E
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
/analysis_layer_2_gdn_2_cond_2_cond_false_197988*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_2_gdn_2_cond_2_cond_true_1979872$
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
?
?
)__inference_analysis_layer_call_fn_196893
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
D__inference_analysis_layer_call_and_return_conditional_losses_1968182
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
%layer_0_gdn_0_cond_1_cond_true_1989041
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
?
?
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_197779H
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
?
v
gdn_0_cond_1_true_199333!
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
?
?
gdn_0_cond_2_cond_true_199425(
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
?
?
%layer_0_gdn_0_cond_2_cond_true_1989878
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
?
?
)analysis_layer_0_gdn_0_cond_2_true_197706I
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
?
w
layer_2_gdn_2_cond_false_1987333
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
?
|
gdn_1_cond_2_true_196406'
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
?
?
/analysis_layer_1_gdn_1_cond_1_cond_false_198193D
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
4analysis_layer_1_gdn_1_cond_1_cond_cond_false_198203*A
output_shapes0
.:,????????????????????????????*F
then_branch7R5
3analysis_layer_1_gdn_1_cond_1_cond_cond_true_1982022)
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
/analysis_layer_0_gdn_0_cond_2_cond_false_198140I
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
?
?
 layer_1_gdn_1_cond_2_true_1991147
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
?
?
C__inference_encoder_layer_call_and_return_conditional_losses_197286

inputs
analysis_197212"
analysis_197214:	?
analysis_197216:	?
analysis_197218#
analysis_197220:
??
analysis_197222
analysis_197224
analysis_197226:	?
analysis_197228
analysis_197230
analysis_197232
analysis_197234#
analysis_197236:
??
analysis_197238:	?
analysis_197240#
analysis_197242:
??
analysis_197244
analysis_197246
analysis_197248:	?
analysis_197250
analysis_197252
analysis_197254
analysis_197256#
analysis_197258:
??
analysis_197260:	?
analysis_197262#
analysis_197264:
??
analysis_197266
analysis_197268
analysis_197270:	?
analysis_197272
analysis_197274
analysis_197276
analysis_197278#
analysis_197280:
??
analysis_197282:	?
identity?? analysis/StatefulPartitionedCall?
 analysis/StatefulPartitionedCallStatefulPartitionedCallinputsanalysis_197212analysis_197214analysis_197216analysis_197218analysis_197220analysis_197222analysis_197224analysis_197226analysis_197228analysis_197230analysis_197232analysis_197234analysis_197236analysis_197238analysis_197240analysis_197242analysis_197244analysis_197246analysis_197248analysis_197250analysis_197252analysis_197254analysis_197256analysis_197258analysis_197260analysis_197262analysis_197264analysis_197266analysis_197268analysis_197270analysis_197272analysis_197274analysis_197276analysis_197278analysis_197280analysis_197282*0
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
D__inference_analysis_layer_call_and_return_conditional_losses_1968182"
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
?
?
!layer_0_gdn_0_cond_1_false_198472-
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
&layer_0_gdn_0_cond_1_cond_false_198481*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_1_cond_true_1984802
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
?
?
*analysis_layer_0_gdn_0_cond_1_false_197624?
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
/analysis_layer_0_gdn_0_cond_1_cond_false_197633*A
output_shapes0
.:,????????????????????????????*A
then_branch2R0
.analysis_layer_0_gdn_0_cond_1_cond_true_1976322$
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
|
gdn_0_cond_2_true_199416'
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
??
?	
"__inference__traced_restore_199921
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
?
?
.analysis_layer_2_gdn_2_cond_1_cond_true_197904C
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
?
?
3analysis_layer_0_gdn_0_cond_1_cond_cond_true_198066K
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
?
?
(analysis_layer_0_gdn_0_cond_false_198037E
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
?
?
7encoder_analysis_layer_0_gdn_0_cond_1_cond_false_195735T
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
<encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_false_195745*A
output_shapes0
.:,????????????????????????????*N
then_branch?R=
;encoder_analysis_layer_0_gdn_0_cond_1_cond_cond_true_19574421
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
?
?
.analysis_layer_2_gdn_2_cond_1_cond_true_198328C
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
<encoder_analysis_layer_2_gdn_2_cond_1_cond_cond_false_196017X
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
?
?
%layer_1_gdn_1_cond_1_cond_true_1986161
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
?
?
(__inference_encoder_layer_call_fn_197515
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
C__inference_encoder_layer_call_and_return_conditional_losses_1974402
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
?
M
gdn_0_cond_true_199322
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
/encoder_analysis_layer_1_gdn_1_cond_true_1958503
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
?
?
.analysis_layer_0_gdn_0_cond_2_cond_true_197715J
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
*layer_1_gdn_1_cond_1_cond_cond_true_1986269
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
3analysis_layer_2_gdn_2_cond_1_cond_cond_true_197914K
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
?
M
gdn_1_cond_true_199466
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
?
?
'analysis_layer_2_gdn_2_cond_true_198308+
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
?
'analysis_layer_0_gdn_0_cond_true_197612+
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
?
?
)analysis_layer_2_gdn_2_cond_2_true_198402I
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
?
?
0encoder_analysis_layer_0_gdn_0_cond_false_195715U
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
?N
?
C__inference_layer_2_layer_call_and_return_conditional_losses_196600

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
gdn_2_cond_false_196477*
output_shapes
: *)
then_branchR
gdn_2_cond_true_1964762

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
gdn_2_cond_1_false_196488*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_1_true_1964872
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
_gradient_op_typeCustomGradient-196533*.
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
_gradient_op_typeCustomGradient-196543*.
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
_gradient_op_typeCustomGradient-196557*$
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
gdn_2_cond_2_false_196571*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_2_cond_2_true_1965702
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
?
?
1encoder_analysis_layer_2_gdn_2_cond_1_true_195997S
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
?
W
gdn_2_cond_false_196477#
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
?
?
&layer_1_gdn_1_cond_1_cond_false_1986172
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
+layer_1_gdn_1_cond_1_cond_cond_false_198627*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_1_gdn_1_cond_1_cond_cond_true_1986262 
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
?
?
&layer_2_gdn_2_cond_1_cond_false_1987532
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
+layer_2_gdn_2_cond_1_cond_cond_false_198763*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_2_gdn_2_cond_1_cond_cond_true_1987622 
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
?
?
gdn_2_cond_1_cond_false_196497"
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
#gdn_2_cond_1_cond_cond_false_196507*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_2_cond_1_cond_cond_true_1965062
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
?
?
"gdn_0_cond_1_cond_cond_true_199352)
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
?
?
&layer_0_gdn_0_cond_1_cond_false_1989052
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
+layer_0_gdn_0_cond_1_cond_cond_false_198915*A
output_shapes0
.:,????????????????????????????*=
then_branch.R,
*layer_0_gdn_0_cond_1_cond_cond_true_1989142 
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
&layer_0_gdn_0_cond_2_cond_false_1989887
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
?
?
!layer_1_gdn_1_cond_1_false_199032-
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
&layer_1_gdn_1_cond_1_cond_false_199041*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_1_gdn_1_cond_1_cond_true_1990402
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
?
?
gdn_1_cond_2_cond_false_196416'
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
"gdn_0_cond_1_cond_cond_true_196178)
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
gdn_2_cond_2_cond_false_196580'
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
?
?
.analysis_layer_0_gdn_0_cond_2_cond_true_198139J
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
?
?
6encoder_analysis_layer_1_gdn_1_cond_1_cond_true_195870S
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
?
?
'analysis_layer_0_gdn_0_cond_true_198036+
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
?
?
%layer_0_gdn_0_cond_1_cond_true_1984801
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
??
?
C__inference_encoder_layer_call_and_return_conditional_losses_198018

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
(analysis_layer_0_gdn_0_cond_false_197613*
output_shapes
: *:
then_branch+R)
'analysis_layer_0_gdn_0_cond_true_1976122
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
*analysis_layer_0_gdn_0_cond_1_false_197624*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_1_true_1976232
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
_gradient_op_typeCustomGradient-197669*.
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
_gradient_op_typeCustomGradient-197679*.
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
_gradient_op_typeCustomGradient-197693*$
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
*analysis_layer_0_gdn_0_cond_2_false_197707*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_0_gdn_0_cond_2_true_1977062
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
(analysis_layer_1_gdn_1_cond_false_197749*
output_shapes
: *:
then_branch+R)
'analysis_layer_1_gdn_1_cond_true_1977482
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
*analysis_layer_1_gdn_1_cond_1_false_197760*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_1_true_1977592
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
_gradient_op_typeCustomGradient-197805*.
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
_gradient_op_typeCustomGradient-197815*.
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
_gradient_op_typeCustomGradient-197829*$
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
*analysis_layer_1_gdn_1_cond_2_false_197843*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_1_gdn_1_cond_2_true_1978422
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
(analysis_layer_2_gdn_2_cond_false_197885*
output_shapes
: *:
then_branch+R)
'analysis_layer_2_gdn_2_cond_true_1978842
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
*analysis_layer_2_gdn_2_cond_1_false_197896*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_1_true_1978952
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
_gradient_op_typeCustomGradient-197941*.
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
_gradient_op_typeCustomGradient-197951*.
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
_gradient_op_typeCustomGradient-197965*$
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
*analysis_layer_2_gdn_2_cond_2_false_197979*A
output_shapes0
.:,????????????????????????????*<
then_branch-R+
)analysis_layer_2_gdn_2_cond_2_true_1979782
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
?
?
gdn_2_cond_1_cond_false_199631"
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
#gdn_2_cond_1_cond_cond_false_199641*A
output_shapes0
.:,????????????????????????????*5
then_branch&R$
"gdn_2_cond_1_cond_cond_true_1996402
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
?
w
layer_2_gdn_2_cond_false_1991573
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
#gdn_2_cond_1_cond_cond_false_199641&
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
?
?
.analysis_layer_1_gdn_1_cond_2_cond_true_197851J
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
?
?
2encoder_analysis_layer_2_gdn_2_cond_2_false_196081U
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
7encoder_analysis_layer_2_gdn_2_cond_2_cond_false_196090*A
output_shapes0
.:,????????????????????????????*I
then_branch:R8
6encoder_analysis_layer_2_gdn_2_cond_2_cond_true_1960892,
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
?
?
0encoder_analysis_layer_1_gdn_1_cond_false_195851U
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
?
?
#gdn_1_cond_1_cond_cond_false_196343&
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
?
W
gdn_2_cond_false_199611#
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
?
|
gdn_2_cond_2_true_196570'
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
?
?
'analysis_layer_1_gdn_1_cond_true_197748+
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
?
?
+layer_1_gdn_1_cond_1_cond_cond_false_1986276
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
?
?
+layer_0_gdn_0_cond_1_cond_cond_false_1984916
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
?
w
layer_0_gdn_0_cond_false_1988853
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
?
?
4analysis_layer_2_gdn_2_cond_1_cond_cond_false_197915H
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
?
v
gdn_2_cond_1_true_196487!
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
??
?
!__inference__wrapped_model_196120
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
0encoder_analysis_layer_0_gdn_0_cond_false_195715*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_0_gdn_0_cond_true_1957142%
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
2encoder_analysis_layer_0_gdn_0_cond_1_false_195726*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_0_gdn_0_cond_1_true_1957252'
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
_gradient_op_typeCustomGradient-195771*.
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
_gradient_op_typeCustomGradient-195781*.
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
_gradient_op_typeCustomGradient-195795*$
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
2encoder_analysis_layer_0_gdn_0_cond_2_false_195809*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_0_gdn_0_cond_2_true_1958082'
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
0encoder_analysis_layer_1_gdn_1_cond_false_195851*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_1_gdn_1_cond_true_1958502%
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
2encoder_analysis_layer_1_gdn_1_cond_1_false_195862*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_1_gdn_1_cond_1_true_1958612'
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
_gradient_op_typeCustomGradient-195907*.
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
_gradient_op_typeCustomGradient-195917*.
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
_gradient_op_typeCustomGradient-195931*$
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
2encoder_analysis_layer_1_gdn_1_cond_2_false_195945*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_1_gdn_1_cond_2_true_1959442'
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
0encoder_analysis_layer_2_gdn_2_cond_false_195987*
output_shapes
: *B
then_branch3R1
/encoder_analysis_layer_2_gdn_2_cond_true_1959862%
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
2encoder_analysis_layer_2_gdn_2_cond_1_false_195998*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_2_gdn_2_cond_1_true_1959972'
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
_gradient_op_typeCustomGradient-196043*.
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
_gradient_op_typeCustomGradient-196053*.
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
_gradient_op_typeCustomGradient-196067*$
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
2encoder_analysis_layer_2_gdn_2_cond_2_false_196081*A
output_shapes0
.:,????????????????????????????*D
then_branch5R3
1encoder_analysis_layer_2_gdn_2_cond_2_true_1960802'
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
?
?
%layer_2_gdn_2_cond_2_cond_true_1992598
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
?N
?
C__inference_layer_1_layer_call_and_return_conditional_losses_196436

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
gdn_1_cond_false_196313*
output_shapes
: *)
then_branchR
gdn_1_cond_true_1963122

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
gdn_1_cond_1_false_196324*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_1_true_1963232
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
_gradient_op_typeCustomGradient-196369*.
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
_gradient_op_typeCustomGradient-196379*.
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
_gradient_op_typeCustomGradient-196393*$
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
gdn_1_cond_2_false_196407*A
output_shapes0
.:,????????????????????????????*+
then_branchR
gdn_1_cond_2_true_1964062
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
?
?
!layer_0_gdn_0_cond_2_false_1985553
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
&layer_0_gdn_0_cond_2_cond_false_198564*A
output_shapes0
.:,????????????????????????????*8
then_branch)R'
%layer_0_gdn_0_cond_2_cond_true_1985632
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
?
w
layer_1_gdn_1_cond_false_1990213
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
trainable_variables
regularization_losses
	variables
	keras_api

signatures
*~&call_and_return_all_conditional_losses
_default_save_signature
?__call__"??
_tf_keras_network??{"name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "name": "analysis", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["analysis", 1, 0]]}, "shared_object_id": 39, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "name": "analysis", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 38}], "input_layers": [["input_1", 0, 0]], "output_layers": [["analysis", 1, 0]]}}}
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
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_sequential??{"name": "analysis", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}}, "beta_initializer": {"class_name": "Ones", "config": {}}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}}}}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null}}]}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, null, null, 3]}, "float32", "lambda_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "AnalysisTransform", "config": {"name": "analysis", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lambda_input"}, "shared_object_id": 1}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 2}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}, "shared_object_id": 3}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 13}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 14}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 23}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 24}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 33}, {"class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 34}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 37}]}}}
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
?
$trainable_variables
%regularization_losses
&	variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAQAAAAIAAABTAAAAcwgAAAB8AGQBGwBTACkCTucAAAAAAOBvQKkAKQHaAXhyAgAA\nAHICAAAA+htibXNoajIwMTgtYXV0b2VuY29kZXItbDIucHnaCDxsYW1iZGE+OgAAAHMAAAAA\n", null, null]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "shared_object_id": 2}
?
(_activation
)_kernel_parameter
_bias_parameter
*trainable_variables
+regularization_losses
,	variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_0", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 3, 192]}, "dtype": "float32"}, "shared_object_id": 3}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 13, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 3]}}
?
._activation
/_kernel_parameter
_bias_parameter
0trainable_variables
1regularization_losses
2	variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_1", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 14}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 23, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
4_activation
5_kernel_parameter
_bias_parameter
6trainable_variables
7regularization_losses
8	variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_2", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": {"class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30}, "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 24}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 33, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
?
:_kernel_parameter
_bias_parameter
;trainable_variables
<regularization_losses
=	variables
>	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "layer_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>SignalConv2D", "config": {"name": "layer_3", "trainable": true, "dtype": "float32", "filters": 192, "kernel_support": {"class_name": "__tuple__", "items": [5, 5]}, "corr": true, "strides_down": {"class_name": "__tuple__", "items": [2, 2]}, "strides_up": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same_zeros", "extra_pad_end": true, "channel_separable": false, "data_format": "channels_last", "activation": "linear", "use_bias": true, "use_explicit": true, "kernel_parameter": {"class_name": "tensorflow_compression>RDFTParameter", "config": {"name": "kernel", "initial_value": null, "dc": true, "shape": {"class_name": "__tuple__", "items": [5, 5, 192, 192]}, "dtype": "float32"}, "shared_object_id": 34}, "bias_parameter": "variable", "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}, "shared_object_id": 35}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 36}, "kernel_regularizer": null, "bias_regularizer": null}, "shared_object_id": 37, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Dlayers
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics
$trainable_variables
%regularization_losses
&	variables
Hlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
I_beta_parameter
J_gamma_parameter
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "gdn_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_0", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 4}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 5}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 6}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 7}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 9}}, "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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

Olayers
Player_regularization_losses
Qnon_trainable_variables
Rmetrics
*trainable_variables
+regularization_losses
,	variables
Slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
T_beta_parameter
U_gamma_parameter
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "gdn_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_1", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 15}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 16}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 17}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 18}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 19}}, "shared_object_id": 20, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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

Zlayers
[layer_regularization_losses
\non_trainable_variables
]metrics
0trainable_variables
1regularization_losses
2	variables
^layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
__beta_parameter
`_gamma_parameter
atrainable_variables
bregularization_losses
c	variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?
{"name": "gdn_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "tensorflow_compression>GDN", "config": {"name": "gdn_2", "trainable": true, "dtype": "float32", "inverse": false, "rectify": false, "data_format": "channels_last", "alpha_parameter": 1.0, "beta_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "beta", "initial_value": null, "minimum": 1e-06, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192]}, "dtype": "float32"}, "shared_object_id": 25}, "gamma_parameter": {"class_name": "tensorflow_compression>GDNParameter", "config": {"name": "gamma", "initial_value": null, "minimum": 0.0, "offset": 3.814697265625e-06, "shape": {"class_name": "__tuple__", "items": [192, 192]}, "dtype": "float32"}, "shared_object_id": 26}, "epsilon_parameter": 1.0, "alpha_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 27}, "beta_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 28}, "gamma_initializer": {"class_name": "Identity", "config": {"gain": 0.1}, "shared_object_id": 8}, "epsilon_initializer": {"class_name": "Ones", "config": {}, "shared_object_id": 29}}, "shared_object_id": 30, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, null, 192]}}
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

elayers
flayer_regularization_losses
gnon_trainable_variables
hmetrics
6trainable_variables
7regularization_losses
8	variables
ilayer_metrics
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
Ktrainable_variables
Lregularization_losses
M	variables
slayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
(0"
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
Vtrainable_variables
Wregularization_losses
X	variables
xlayer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
.0"
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
atrainable_variables
bregularization_losses
c	variables
}layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
C__inference_encoder_layer_call_and_return_conditional_losses_197129
C__inference_encoder_layer_call_and_return_conditional_losses_197206
C__inference_encoder_layer_call_and_return_conditional_losses_198018
C__inference_encoder_layer_call_and_return_conditional_losses_198442?
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
!__inference__wrapped_model_196120?
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
(__inference_encoder_layer_call_fn_197361
(__inference_encoder_layer_call_fn_197515?
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
D__inference_analysis_layer_call_and_return_conditional_losses_196647
D__inference_analysis_layer_call_and_return_conditional_losses_196734
D__inference_analysis_layer_call_and_return_conditional_losses_198866
D__inference_analysis_layer_call_and_return_conditional_losses_199290?
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
)__inference_analysis_layer_call_fn_196893
)__inference_analysis_layer_call_fn_197051?
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
$__inference_signature_wrapper_197594input_1"?
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
?2?
B__inference_lambda_layer_call_and_return_conditional_losses_199296
B__inference_lambda_layer_call_and_return_conditional_losses_199302?
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
?2?
C__inference_layer_0_layer_call_and_return_conditional_losses_199446?
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
C__inference_layer_1_layer_call_and_return_conditional_losses_199590?
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
C__inference_layer_2_layer_call_and_return_conditional_losses_199734?
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
C__inference_layer_3_layer_call_and_return_conditional_losses_199752?
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
!__inference__wrapped_model_196120?:??????????????????????J?G
@?=
;?8
input_1+???????????????????????????
? "N?K
I
analysis=?:
analysis,?????????????????????????????
D__inference_analysis_layer_call_and_return_conditional_losses_196647?:??????????????????????W?T
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
D__inference_analysis_layer_call_and_return_conditional_losses_196734?:??????????????????????W?T
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
D__inference_analysis_layer_call_and_return_conditional_losses_198866?:??????????????????????Q?N
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
D__inference_analysis_layer_call_and_return_conditional_losses_199290?:??????????????????????Q?N
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
)__inference_analysis_layer_call_fn_196893?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p

 
? "3?0,?????????????????????????????
)__inference_analysis_layer_call_fn_197051?:??????????????????????W?T
M?J
@?=
lambda_input+???????????????????????????
p 

 
? "3?0,?????????????????????????????
C__inference_encoder_layer_call_and_return_conditional_losses_197129?:??????????????????????R?O
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
C__inference_encoder_layer_call_and_return_conditional_losses_197206?:??????????????????????R?O
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
C__inference_encoder_layer_call_and_return_conditional_losses_198018?:??????????????????????Q?N
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
C__inference_encoder_layer_call_and_return_conditional_losses_198442?:??????????????????????Q?N
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
(__inference_encoder_layer_call_fn_197361?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p

 
? "3?0,?????????????????????????????
(__inference_encoder_layer_call_fn_197515?:??????????????????????R?O
H?E
;?8
input_1+???????????????????????????
p 

 
? "3?0,?????????????????????????????
B__inference_lambda_layer_call_and_return_conditional_losses_199296?Q?N
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
B__inference_lambda_layer_call_and_return_conditional_losses_199302?Q?N
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
C__inference_layer_0_layer_call_and_return_conditional_losses_199446????????I?F
??<
:?7
inputs+???????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_1_layer_call_and_return_conditional_losses_199590????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_2_layer_call_and_return_conditional_losses_199734????????J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
C__inference_layer_3_layer_call_and_return_conditional_losses_199752??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
$__inference_signature_wrapper_197594?:??????????????????????U?R
? 
K?H
F
input_1;?8
input_1+???????????????????????????"N?K
I
analysis=?:
analysis,????????????????????????????