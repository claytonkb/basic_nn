basic_nn
========

A single hidden-layer neural network in a single C-file. Hat-tip to 
[glouw/tinn](https://github.com/glouw/tinn).

License: Public Domain

I built basic_nn.c with the following command (don't forget to link math-library
with -lm).

    gcc -O3 -o basic_nn basic_nn.c -lm

You need to have a file named "semeion.data" in the run directory or basic_nn
will exit with an error. I have included a stripped version of the dataset in
the repository (850k).

Once basic_nn is at the prompt, the neural net is initialized (with random
weights) and the training dataset is loaded. Enter 0 to get the menu of
commands. You can export each digit in the dataset into an SVG file using the
command codes 51 (training) and 52 (test), followed by the line-number of the
file for the digit you want to export. You can inspect the internal weights and
states of the neural network in SVG format in a similar fashion (3x and 4x
codes).

You can iteratively train the network with command code 8. You can export an
SVG snapshot of any part of the neural network as you train it. You can compare
the outputs of the neural network with the training data with the following
command code sequence: 

    % 72 x   <== drive the xth test pair input into the NN
    % 33     <== display the output states of the NN (actual values)
    % 62 x   <== display the xth test pair outputs (expected values)
    % 92 x   <== measure the error of the NN outputs w.r.t the xth test pair output

... where x is the line-number in the data-set for the character you want to
test. For example, after training the network for 20 iterations, I compared
test-pair 197 expected versus actual (match notated):

    % 8 20
    % 72 197
    % 33
    0.001765
    0.000000
    0.000017
    0.000039
    0.609137 <<< MATCH!!
    0.000085
    0.086282
    0.000002
    0.000000
    0.000000
    % 62 197
    0.000000 0.000000 0.000000 0.000000 1.000000 0.000000 0.000000 0.000000 0.000000 0.000000 
    % 92 197                            ^^^MATCH!!
    error 0.080111

You can also check the overall global error of the network with command code
93. Immediately after start-up, the weights are untrained and the error rate
will be high:

    % 93
    num_failures/TEST_SET_SIZE 1.000000

By training for one iteration, we can bring the error down (by a lot):

    % 8 1
    % 93
    num_failures/TEST_SET_SIZE 0.316583

Training 20 more iterations brings it further down:

    % 8 20
    % 93
    num_failures/TEST_SET_SIZE 0.130653


