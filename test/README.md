Instruction
===========

1. Pull the docker image having PyTorch-1.1.0 installed from source:
~~~bash
docker pull hibagus/superscalar-project:pytorch-1.1.0
~~~
Or build the image from the [Dockerfile](Dockerfile) in this folder:
~~~bash
docker build . -t hibagus/superscalar-project:pytorch-1.1.0
~~~

2. Launch a container with this repository amounted:
~~~bash
docker run -dti --name convglu_test -v $PWD/../../FusedConvGLU:/convglu hibagus/superscalar-project:pytorch-1.1.0
~~~

3. Log into the container:
~~~bash
docker exec -ti convglu_test /bin/bash
~~~
The following steps will be completed within the container.

4. Go to the mounted folder having your implemented extension, for example:
~~~bash
cd /convglu/cpp
~~~

5. Compile and install the extension with the PyTorch's tool:
~~~bash
python setup.py install
~~~
A quick check of your extension module is successfully installed is to execute the following command in a python interpreter (REPL):
~~~python
import torch
import convtbcglu
help(convtbcglu.forward)
~~~
It should print out the method's signature, which matches what you defined in the C++ code.

6. Run the test script:
~~~bash
./convglu.py --loss testloss testin
~~~
The expected output is in the file `testout`.
If the output file, `test.tsv`, you generated just now,
has no difference with `testout`,
then congratulations, you get the forward implemented correctly.  
The expected gradiant is in the file `testgrads`.
If the output gradiant file, `test_grads.tsv`, you generated just now,
has no different with `testgrads`,
then congratulations again, you get the backward implemented right as well!
