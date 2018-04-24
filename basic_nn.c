// basic_nn.c
//
// Basic single hidden-layer neural network in one file

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "cutils.h"

#define random_wt() ((rand() / (float) RAND_MAX) - 0.5f)

typedef float activation_fn(float);

typedef struct{ // array of floats
    float *f;
    int    n;
} farray;

typedef struct{ // weight matrix
    farray     wts; // array containing weights
    int     in_dim; // incoming (to this layer)   dimension
    int    out_dim; // outgoing (from this layer) dimension
} wt_mat;

typedef struct{ // neural net layer
    wt_mat in_wts; // incoming weights to this layer
    farray states; // states for this layer; assert(in_wts.out_dim == states.n)
    float  bias;
} nn_layer;

typedef struct{ // 1-hidden layer NN
    nn_layer hidden;
    nn_layer out;
    activation_fn *s;
} hid1_nn;

typedef struct{
    farray in;
    farray out;
} training_pair;

typedef struct{
    training_pair  **pairs;
    int           n_pairs;
} training_set;


hid1_nn *construct_nn(int num_inputs, int num_hidden, int num_outputs, activation_fn *s, int init);
void randomize_wts(hid1_nn *nn);
void back_propagate(hid1_nn *nn, farray *in, farray *target, float rate);
void forward_propagate(hid1_nn *nn, farray *in);
float logistic(float f);
float delta_logistic(float f);
float ms_error(float a, float b);
float delta_error(float a, float b);
void dev_prompt(void);
void dev_get_line(char *buffer, FILE *stream);
void dev_menu(void);
void show_states(nn_layer *layer);
void show_weights(nn_layer *layer);
void introspect_svg(farray *arr, unsigned width, unsigned height, float offset, FILE *stream);
float sum_error(float *targets, float *outputs, int num_outputs);
FILE *open_file(char *filename, char *attr);
int   file_size(FILE *file);
char *slurp_file(char *filename);
training_set *load_semeion_data(void);
float nn_train(hid1_nn *nn, farray *in, farray *target, float rate);
void nn_predict(hid1_nn *nn, farray *in, farray *target);


//
//
int main(void){

    srand(time(0));

    dev_prompt();

    return 0;

}


//
//
void back_propagate(hid1_nn *nn, farray *in, farray *target, float rate){

    int i,j;
    float sum,a,b;

    nn_layer *hidden = &nn->hidden;
    nn_layer *out    = &nn->out;
    activation_fn *s = nn->s;

    int num_inputs  = hidden->in_wts.in_dim; // -1 for bias
    int num_hidden  =    out->in_wts.in_dim; // -1 for bias
    int num_outputs =    out->in_wts.out_dim;

    float *wih = nn->hidden.in_wts.wts.f;
    float *who = nn->out.in_wts.wts.f;

    for(i = 0; i < num_hidden; i++){

        sum=0;

        for(j = 0; j < num_outputs; j++){

            a = delta_error(out->states.f[j], target->f[j]);
            b = delta_logistic(out->states.f[j]);
            sum += a * b * who[j*num_hidden + i];
            who[j*num_hidden + i] = who[j*num_hidden + i] - (rate * a * b * hidden->states.f[i]);

        }

        for(j = 0; j < num_inputs; j++){
            wih[i*num_inputs + j] = wih[i*num_inputs + j] - (rate * sum * in->f[j] * delta_logistic(hidden->states.f[i]));
        }

    }

}


//
//
void forward_propagate(hid1_nn *nn, farray *in){

    int i,j;
    float sum;

    nn_layer *hidden = &nn->hidden;
    nn_layer *out    = &nn->out;
    activation_fn *s = nn->s;

    int num_inputs  = hidden->in_wts.in_dim;
    int num_hidden  = out->in_wts.in_dim;
    int num_outputs = out->in_wts.out_dim;

    float *wih = nn->hidden.in_wts.wts.f;
    float *who = nn->out.in_wts.wts.f;

    for(i=0; i < num_hidden; i++){
        sum=0;
        for(j=0; j < num_inputs; j++)
            sum += in->f[j] * wih[i*num_inputs + j];
        hidden->states.f[i] = s(sum+hidden->bias);
    }

    for(i=0; i < num_outputs; i++){
        sum=0;
        for(j=0; j < num_hidden; j++)
            sum += hidden->states.f[j] * who[i*num_hidden + j];
        out->states.f[i] = s(sum+out->bias);
    }

}



//
//
float logistic(float x){
    return 1.0f / (1.0f + expf(-1*x));
}


// closed-form derivative of logistic function
//
float delta_logistic(float x){
    return x * (1.0f - x);
}


// mean-squared error
//
float ms_error(float a, float b){
    float delta = a-b;
    return ((delta*delta) / 2);
}


// closed-form derivative of the mean-squared error
//
float delta_error(float a, float b){
    return a-b;
}


//
//
float sum_error(float *targets, float *outputs, int num_outputs){
    float sum = 0.0f;
    for(int i = 0; i<num_outputs; i++)
        sum += ms_error(targets[i], outputs[i]);
    return sum;
}


//
//
void show_states(nn_layer *layer){
    int i;
    for(i=0; i < layer->states.n; i++){
        printf("%f\n",layer->states.f[i]);
    }
}


//
//
void show_weights(nn_layer *layer){
    int i;
    for(i=0; i < layer->in_wts.wts.n; i++){
        printf("%f\n",layer->in_wts.wts.f[i]);
    }
}


//
//
hid1_nn *construct_nn(int num_inputs, int num_hidden, int num_outputs, activation_fn *s, int init){

    //Allocate stuff
    hid1_nn *nn = malloc(sizeof(hid1_nn));

    nn_layer *hidden = &nn->hidden;
    nn_layer *out    = &nn->out;

    hidden->in_wts.wts.n   = (num_inputs)*num_hidden; // +1 for bias
    hidden->in_wts.wts.f   = malloc(hidden->in_wts.wts.n * sizeof(float));
    hidden->in_wts.in_dim  = num_inputs; // +1 for bias
    hidden->in_wts.out_dim = num_hidden;
    hidden->states.f       = malloc(num_hidden*sizeof(float));
    hidden->states.n       = num_hidden;
    hidden->bias           = random_wt();

    out->in_wts.wts.n      = (num_hidden)*num_outputs; // +1 for bias
    out->in_wts.wts.f      = malloc(out->in_wts.wts.n * sizeof(float));
    out->in_wts.in_dim     = num_hidden;
    out->in_wts.out_dim    = num_outputs;
    out->states.f          = malloc(num_outputs*sizeof(float));
    out->states.n          = num_outputs;
    out->bias              = random_wt();

    nn->s = s;

    if(init){
        randomize_wts(nn);
    }    

    return nn;

}


//
//
void randomize_wts(hid1_nn *nn){

    int i;
    float *wt_array;
    farray *hidden_wts = &(nn->hidden.in_wts.wts);
    farray *out_wts    = &(nn->out.in_wts.wts);

    wt_array = hidden_wts->f;
    for(i=0; i < hidden_wts->n; i++){
        wt_array[i] = random_wt();
    }

    wt_array = out_wts->f;
    for(i=0; i < out_wts->n; i++){
        wt_array[i] = random_wt();
    }

}


//
//
FILE *open_file(char *filename, char *attr){

    FILE* file;

    file = fopen((char*)filename, attr);

    if(file==NULL)
        _fatal((char*)filename);

    return file;

}


//
//
int file_size(FILE *file){

    fseek(file, 0L, SEEK_END);
    int size = ftell(file);
    rewind(file);

    return size;

}


//
//
char *slurp_file(char *filename){

    FILE *f = open_file((char*)filename, "r");
    int size = file_size(f);

    char *file_buffer = malloc(size+1);
    size_t dummy = fread((char*)file_buffer, 1, size, f);

    fclose(f);

    return file_buffer;

}


//
//
training_set *load_semeion_data(void){

    #define TRAINING_SET_SIZE    1593
    #define TRAINING_INPUT_SIZE  256
    #define TRAINING_OUTPUT_SIZE 10
    #define TRAINING_SET_FILE    "semeion.data"

    char *training_str = slurp_file(TRAINING_SET_FILE);

    int i,j=0;
    int line_count;
    char c;

    training_set *ts = malloc(sizeof(training_set));
    ts->pairs   = malloc(TRAINING_SET_SIZE * sizeof(training_pair*));
    ts->n_pairs = TRAINING_SET_SIZE;

    int writes=0;

    for(i=0; i<TRAINING_SET_SIZE; i++){
        ts->pairs[i]      = malloc(sizeof(training_pair));
        ts->pairs[i]->in.f  = malloc(TRAINING_INPUT_SIZE * sizeof(float));
        ts->pairs[i]->in.n  = TRAINING_INPUT_SIZE;
        ts->pairs[i]->out.f = malloc(TRAINING_OUTPUT_SIZE * sizeof(float));
        ts->pairs[i]->out.n = TRAINING_OUTPUT_SIZE;

        c=training_str[j];
        line_count=0;
        while(c != '\0' && c != '\n' && (line_count < 256)){
            if(c=='0' || c=='1'){
                ts->pairs[i]->in.f[line_count++] = (float)(c-'0');
            }
            c=training_str[++j];
        }
        line_count=0;
        while(c != '\0' && c != '\n' && (line_count < 10)){
            if(c=='0' || c=='1'){
                ts->pairs[i]->out.f[line_count++] = (float)(c-'0');
            }
            c=training_str[++j];
        }
        while(c != '0' && c != '1'){
            c=training_str[++j];
        }
    }

    return ts;

}


//
//
void introspect_svg(farray *arr, unsigned width, unsigned height, float offset, FILE *stream){

    #define SVG_RECT_SCALE 20

    unsigned svg_height = height*SVG_RECT_SCALE;
    unsigned svg_width  = width *SVG_RECT_SCALE;

    unsigned array_length = arr->n;//array8_size(arr);

    int i,j,ctr=0;
    long int z;
    unsigned char byte;

    fprintf(stream,
        "<svg width=\"%d\" height=\"%d\" viewBox=\"0 0 %d %d\" xmlns=\"http://www.w3.org/2000/svg\">\n",
        svg_width,
        svg_height,
        svg_width,
        svg_height);

    for(i=0;i<height;i++){
        for(j=0;j<width;j++){

            if(ctr++ >= array_length) break;

            z = lrintf(255*(arr->f[i*width+j] + offset));
            byte = (unsigned char)z;

            fprintf(stream,
                "<rect x=\"%d\" y=\"%d\" width=\"%d\" height=\"%d\" stroke=\"none\" fill=\"#%02x%02x%02x\"/>\n",
                (SVG_RECT_SCALE*j), (SVG_RECT_SCALE*i),
                 SVG_RECT_SCALE,     SVG_RECT_SCALE,
                byte, byte, byte);

        }
    }

    fprintf(stream, "</svg>\n\n");

}


//
//
void dev_prompt(void){

    hid1_nn *nn = construct_nn(256, 32, 10, logistic, 1);
    training_set *ts = load_semeion_data();

    char *cmd_code_str;
    int   cmd_code=0;

    char buffer[256];
    int i,j,k;
    float ftemp;
    float total_error;
    FILE *s;

    farray q;
    farray z;

    q.f = ts->pairs[0]->in.f;
    q.n = 256;

    z.f = ts->pairs[0]->out.f;
    z.n = 10;

    _say("type 0 for menu");

    while(1){

        _prn("% ");

        dev_get_line(buffer, stdin);

        cmd_code_str = strtok(buffer, " ");
        if(cmd_code_str == NULL) continue;
        cmd_code = atoi(cmd_code_str);

        switch(cmd_code){
            case 0:
                dev_menu();
                break;
            case 1:
                _say("cmd_code=1");
                break;
            case 2:
                _say("exiting");
                return;
            case 31:
                show_states(&(nn->hidden));
                break;
            case 32:
                show_weights(&(nn->hidden));
                break;
            case 33:
                show_states(&(nn->out));
                break;
            case 34:
                show_weights(&(nn->out));
                break;
            case 41:
                s = fopen("test.svg", "wb");
                introspect_svg( &(nn->hidden.states),
                                  nn->hidden.states.n, 
                                  1, 0.5, s);
                fclose(s);
                break;
            case 42:
                s = fopen("test.svg", "wb");
                introspect_svg( &(nn->hidden.in_wts.wts),
                                  nn->hidden.in_wts.in_dim, 
                                  nn->hidden.in_wts.out_dim, 
                                  0.5, s);
                fclose(s);
                break;
            case 43:
                s = fopen("test.svg", "wb");
                introspect_svg( &(nn->out.states),
                                  nn->out.states.n, 
                                  1, 0.5, s);
                fclose(s);
                break;
            case 44:
                s = fopen("test.svg", "wb");
                introspect_svg( &(nn->out.in_wts.wts),
                                  nn->out.in_wts.in_dim, 
                                  nn->out.in_wts.out_dim, 
                                  0.5, s);
                fclose(s);
                break;
            case 5:
                cmd_code_str = strtok(NULL, " ");
                if(cmd_code_str == NULL){ _say("no argument given"); continue; }
                cmd_code = atoi(cmd_code_str);
                q.f = ts->pairs[cmd_code]->in.f;
                s = fopen("test.svg", "wb");
                introspect_svg(&q, 16, 16, 0, s);
                fclose(s);
                break;
            case 6:
                cmd_code_str = strtok(NULL, " ");
                if(cmd_code_str == NULL){ _say("no argument given"); continue; }
                cmd_code = atoi(cmd_code_str);
                for(i=0; i < nn->out.states.n; i++){
                    printf("%f ", ts->pairs[cmd_code]->out.f[i]);
                }
                printf("\n");
                break;
            case 71:
                cmd_code_str = strtok(NULL, " ");
                if(cmd_code_str == NULL){ _say("no argument given"); continue; }
                cmd_code = atoi(cmd_code_str);
                q.f = ts->pairs[cmd_code]->in.f;
                forward_propagate(nn,&q);
                break;
            case 72:
                cmd_code_str = strtok(NULL, " ");
                if(cmd_code_str == NULL){ _say("no argument given"); continue; }
                cmd_code = atoi(cmd_code_str);
                q.f = ts->pairs[cmd_code]->in.f;
                z.f = ts->pairs[cmd_code]->out.f;
                back_propagate(nn,&q,&z,1);
                break;
            case 8:
                cmd_code_str = strtok(NULL, " ");
                if(cmd_code_str == NULL){ _say("no argument given"); continue; }
                cmd_code = atoi(cmd_code_str);
                for(j=0;j<cmd_code;j++){
                    for(i=0;i<1593;i++){
                        ftemp=(1592*( (float)rand() / RAND_MAX));
                        k=floor(ftemp);
                        q.f = ts->pairs[k]->in.f;
                        z.f = ts->pairs[k]->out.f;
                        forward_propagate(nn,&q);
                        back_propagate(nn,&q,&z,1);
                        q.f = ts->pairs[i]->in.f;
                        z.f = ts->pairs[i]->out.f;
                        forward_propagate(nn,&q);
                        back_propagate(nn,&q,&z,1);
                    }
                }
                break;
            case 9:
                total_error=0;
                for(i=0;i<1593;i++){
                    q.f = ts->pairs[i]->in.f;
                    forward_propagate(nn,&q);
                    total_error += 
                        sum_error(
                            ts->pairs[i]->out.f,
                            nn->out.states.f,
                            nn->out.states.n);
                }
                _df(total_error/1593);
                break;
            default:
                _say("unrecognized cmd_code");
                dev_menu();
                break;
        }

        for(i=0;i<256;i++){ buffer[i]=0; } // zero out the buffer

    }

}


//
//
void dev_get_line(char *buffer, FILE *stream){

    int c, i=0;

    while(1){ //FIXME unsafe, wrong
        c = fgetc(stream);
        if(c == EOF || c == '\n'){
            break;
        }
        buffer[i] = c;
        i++;
    }

    buffer[i] = '\0';

}


//
//
void dev_menu(void){

    _say( "\n0     .....    list command codes\n"
            "1     .....    dev one-off\n"
            "2     .....    exit\n"
            "31    .....    show hidden states\n"
            "32    .....    show hidden weights\n"
            "33    .....    show output states\n"
            "34    .....    show output weights\n"
            "41    .....    SVG  hidden states     ==> test.svg\n"
            "42    .....    SVG  hidden weights    ==> test.svg\n"
            "43    .....    SVG  output states     ==> test.svg\n"
            "44    .....    SVG  output weights    ==> test.svg\n"
            "5  x  .....    SVG  training input x  ==> test.svg\n"
            "6  x  .....    show training output x ==> test.svg\n"
            "71 x  .....    fwd prop train pair x\n"
            "72 x  .....    back prop train pair x\n"
            "8  x  .....    x iters auto training\n"
            "9     .....    global error\n");

}


// Clayton Bauman 2018

