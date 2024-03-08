#include <iostream>
#include <math.h>


#define ARR_LEN(x) (sizeof(x) / sizeof(*x))



double randd(double from, double to)
{
    return ((double)rand() / (double)RAND_MAX) * (to - from) + from;
}


double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}




struct NN
{
    NN();
    NN(size_t* layers, size_t layers_len);


    size_t* layers;
    size_t layers_len;

    double** biases;
    double** weights;


    void forward(double* output, double* input);
};

NN::NN()
{
    layers_len = 0;
    layers = nullptr;
    biases = nullptr;
    weights = nullptr;
}

NN::NN(size_t* layers, size_t layers_len) : layers(layers), layers_len(layers_len)
{
    biases = (double**)malloc((layers_len - 1) * sizeof(*biases));
    for (size_t i = 0; i < layers_len - 1; ++i)
        biases[i] = (double*)malloc((layers[i+1]) * sizeof(**biases));


    weights = (double**)malloc((layers_len - 1) * sizeof(*weights));
    for (size_t i = 0; i < layers_len - 1; ++i)
        weights[i] = (double*)malloc((layers[i+1] * layers[i]) * sizeof(**weights));


    for (size_t i = 0; i < layers_len - 1; ++i)
    {
        for (size_t j = 0; j < layers[i+1]; ++j)
            biases[i][j] = randd(-10.0, 10.0);

        for (size_t j = 0; j < (layers[i+1] * layers[i]); ++j)
            weights[i][j] = randd(-10.0, 10.0);
    }
}

void NN::forward(double* output, double* input)
{
    size_t max_layer = 0;
    for (size_t i = 0; i < layers_len; ++i) 
        if (layers[i] > max_layer)
            max_layer = layers[i];
    
    double** layer_buffer = (double**)malloc(2 * sizeof(*layer_buffer));
    layer_buffer[0] = (double*)malloc(max_layer * sizeof(**layer_buffer));
    layer_buffer[1] = (double*)malloc(max_layer * sizeof(**layer_buffer));


    for (size_t i = 0; i < layers[0]; ++i)
        layer_buffer[0][i] = input[i];

    for (size_t i = 1; i < layers_len; ++i) {
        for (size_t nli = 0; nli < layers[i]; ++nli) { // nli -> next layer index

            for (size_t pli = 0; pli < layers[i-1]; ++pli) { // pli -> previous layer index
                layer_buffer[i%2][nli] += layer_buffer[(i-1)%2][pli] * weights[i-1][nli * layers[i-1] + pli];
            }

            layer_buffer[i%2][nli] += biases[i-1][nli];

            layer_buffer[i%2][nli] = sigmoid(layer_buffer[i%2][nli]);
        }
    }


    for (size_t i = 0; i < layers[layers_len-1]; ++i)
        output[i] = layer_buffer[(layers_len-1) % 2][i];

    free(layer_buffer[0]);
    free(layer_buffer[1]);
    free(layer_buffer);

    
}


double error(NN* nn, double* in, double* out)
{
    double err = 0.0;
    
    double* out_predicted = (double*)malloc(nn->layers[nn->layers_len-1] * sizeof(*out_predicted));
    nn->forward(out_predicted, in);

    for (size_t i = 0; i < nn->layers[(nn->layers_len)-1]; ++i)
        err += (out[i] - out_predicted[i]) * (out[i] - out_predicted[i]);
    
    free(out_predicted);

    return err;
}


void NNcpy(NN* to, NN* from)
{
    *to = NN(from->layers, from->layers_len);


    for (size_t layer = 0; layer < to->layers_len - 1; ++layer)
        for (size_t in_layer = 0; in_layer < to->layers[layer]; ++in_layer)
        {
            to->weights[layer][in_layer] = from->weights[layer][in_layer];

            to->biases[layer][in_layer] = from->biases[layer][in_layer];
        }

}

void gradient_descent(NN* nn, double* in, double* out, double delta, double k)
{
    NN bnn; // bnn -> buffer neural network
    NNcpy(&bnn, nn);
    
    NN derr = NN(nn->layers, nn->layers_len); // derr -> delta error nn



    double err = error(nn, in, out);

    for (size_t layer = 0; layer < nn->layers_len - 1; ++layer)
    {
        for (size_t in_layer = 0; in_layer < nn->layers[layer]; ++in_layer)
        {
            bnn.weights[layer][in_layer] += delta;
            derr.weights[layer][in_layer] = error(&bnn, in, out) - err;
            bnn.weights[layer][in_layer] = nn->weights[layer][in_layer];

            bnn.biases[layer][in_layer] += delta;
            derr.biases[layer][in_layer] = error(&bnn, in, out) - err;
            bnn.biases[layer][in_layer] = nn->biases[layer][in_layer];
        }
    }


    for (size_t layer = 0; layer < nn->layers_len - 1; ++layer)
    {
        for (size_t in_layer = 0; in_layer < nn->layers[layer]; ++in_layer)
        {
            nn->weights[layer][in_layer] -= derr.weights[layer][in_layer] * k;
            nn->biases[layer][in_layer] -= derr.biases[layer][in_layer] * k;
        }
    }

    printf("%5lf --> %5lf (de = %5lf)\n", err, error(nn, in, out), error(nn, in, out) - err);

}




double in[][2] = 
{
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0},
};

double out[][1] = 
{
    {0.0},
    {0.0},
    {0.0},
    {1.0},
};


int main(void) {

    size_t layers[] = {2, 1};

    NN nn(layers, ARR_LEN(layers));


    for (size_t iterations = 0; iterations < 20000; ++iterations)
        for (size_t i = 0; i < 4; ++i)
        {
            gradient_descent(&nn, in[i], out[i], 0.5, 0.5);
        }

}
