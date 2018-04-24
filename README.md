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
command code 5, followed by the line-number of the file for the digit you want
to export. You can inspect the internal weights and states of the neural
network in SVG format in a similar fashion.

You can iteratively train the network with command code 8. You can export an
SVG snapshot of any part of the neural network as you train it. You can compare
the outputs of the neural network with the training data with the following
command code sequence: 

    % 71 x
    % 33
    % 6 x

... where x is the line-number in the data-set for the character you want to
test. For example, after training the network for 20 iterations, I compared
line 197 expected versus actual (match notated):

    % 8 20
    % 71 197
    % 33
    0.000142
    0.000757
    0.000554
    0.001212
    0.000063
    0.000082
    0.000000
    0.000028
    0.000322
    0.999582  <<< MATCH!
    % 6 197
    0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 1.000000 
                                                                                     ^^^^^^^^
                                                                                     MATCH!

You can also check the overall global error of the network with command code
9. Immediately after start-up, the weights are untrained and the error rate
will be high:

    % 9
    total_error/1593 1.227026

By training for one iteration, we can bring the error down (by a lot):

    % 8 1
    % 9
    total_error/1593 0.095878

Training 20 more iterations brings it further down:

    % 8 20
    % 9
    total_error/1593 0.001405




