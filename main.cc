#include "stack.h"

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

struct FixedArrayF32
{
    float* data;
    int size;
};

struct FixedArrayI32
{
    int* data;
    int size;
};

struct VectorF32
{
    float* data;
    int size;
};

struct MatrixF32
{
    float* data;
    int row;
    int column;
};

struct NeuralNetwork
{
    // the neuron number from input layer to output layer
    FixedArrayI32 layer;
    // the neuron output value, store in a top to bottom order, from the input layer to the output layer
    FixedArrayF32 neuron;
    // the weight stored in a top to bottom order of the neuron in the follwing layer,
    // which is all the weight from l-1 layer to the topmost neuron in layer l first,
    // then all the weight from l-1 layer to the second top neuron in layer l, and so on
    FixedArrayF32 weight;
    // the bias stored in the same order as the neuron
    FixedArrayF32 bias;
};

int mnist_read_int(FILE* fp)
{
    assert(fp != NULL);
    uint8_t buffer[4];
    fread(buffer, sizeof(uint8_t), 4, fp);
    int result =
        buffer[0] << 24 | buffer[1] << 16 | buffer[2] << 8 | buffer[3];
    return result;
}

int main(void)
{
    StackAllocator stack = {};
    void* buffer = malloc(MB(64));
    stack_init(&stack, buffer, MB(64));

    // read mnist data
    FILE* fp;

    // training image data
    fp = fopen("mnist/train-images.idx3-ubyte", "rb");
    assert(fp != NULL);

    const int train_images_data_magic_number = mnist_read_int(fp);
    assert(train_images_data_magic_number == 2051);
    const int train_images_total = mnist_read_int(fp);
    const int train_image_pixel_rows = mnist_read_int(fp);
    const int train_image_pixel_columns = mnist_read_int(fp);

    const int train_image_size = train_image_pixel_rows * train_image_pixel_columns;
    assert(train_image_size == 784);
    const int train_images_data_size = train_image_size * train_images_total;
    const uint8_t* train_images_data =
        (uint8_t*)stack_alloc(&stack, train_images_data_size);
    assert(train_images_data != NULL);
    const size_t train_images_data_read_byte = 
        fread((void*)train_images_data, sizeof(uint8_t), train_images_data_size, fp);
    assert(train_images_data_read_byte == train_images_data_size);

    fclose(fp);

    // training label data
    fp = fopen("mnist/train-labels.idx1-ubyte", "rb");
    assert(fp != NULL);

    const int train_labels_magic_number = mnist_read_int(fp);
    assert(train_labels_magic_number == 2049);
    const int train_labels_total = mnist_read_int(fp);
    assert(train_labels_total == train_images_total);
    const uint8_t* train_labels_data =
        (uint8_t*)stack_alloc(&stack, train_labels_total * sizeof(uint8_t));
    assert(train_labels_data != NULL);

    const size_t train_labels_data_read_byte =
        fread((void*)train_labels_data, sizeof(uint8_t), train_labels_total, fp);
    assert(train_labels_data_read_byte == train_labels_total);

    fclose(fp);

    // neural network define
    // this is a two layer (or three layer if you count input layer in) neural network
    // layer 0: input layer, 784 neuron (each neuron represent a pixel of image)
    // layer 1: hidden layer, 30 neuron
    // layer 2: output layer, 10 neuron
    NeuralNetwork nn{};
    {
        nn.layer.size = 3;
        nn.layer.data = (int*)stack_alloc(&stack, nn.layer.size * sizeof(int));
        nn.layer.data[0] = train_image_size;
        nn.layer.data[1] = 30;
        nn.layer.data[2] = 10;

        int total_neuron = 0;
        for (int i = 0; i < nn.layer.size; i++)
        {
            total_neuron += nn.layer.data[i];
        }
        nn.neuron.size = total_neuron;
        nn.neuron.data = (float*)stack_alloc(&stack, nn.neuron.size * sizeof(float));

        int total_weight = 0;
        for (int i = 1; i < nn.layer.size; i++)
        {
            int prev = nn.layer.data[i - 1];
            int next = nn.layer.data[i];
            total_weight += (prev * next);
        }
        nn.weight.size = total_weight;
        nn.weight.data = (float*)stack_alloc(&stack, nn.weight.size * sizeof(float));

        int total_bias = 0;
        for (int i = 1; i < nn.layer.size; i++)
        {
            total_bias += nn.layer.data[i];
        }
        nn.bias.size = total_bias;
        nn.bias.data = (float*)stack_alloc(&stack, nn.bias.size * sizeof(float));

        // using random number initialize weights and initialize all bias to 0
        // TODO: using xavier initialization ?
        srand(time(NULL));
        for (int i = 0; i < nn.weight.size; i++)
        {
            // generate a random number between [-1.0, 1.0]
            float random = 2 * ((float)rand() / (float)RAND_MAX) - 1.0f;
            nn.weight.data[i] = random;
        }
        memset(nn.bias.data, 0, nn.bias.size * sizeof(float));
    }

    clock_t training_start = clock();

    // training
    {
        // some training setting
        const float learning_rate = 3.0f;
        const int total_epochs = 30;
        const int mini_batch_size = 10;
        const int training_per_epoch =
            train_images_total % mini_batch_size == 0
            ? train_images_total / mini_batch_size
            : train_images_total / mini_batch_size + 1;

        // indices of the training image dataset, we will shuffle indices before each epoch
        FixedArrayI32 training_image_index{};
        training_image_index.size = train_images_total;
        training_image_index.data = (int*)stack_alloc(&stack, training_image_index.size * sizeof(int));
        for (int i = 0; i < train_images_total; i++)
        {
            training_image_index.data[i] = i;
        }

        // Z is the weighted sum of a neuron's input 
        // Z = w*x + b; (w is weight, b is bias, x is input)
        // Output is the output of a neuron, calculated from the activation function
        // Output = Activation(Z);
        // Error is the loss/cost value of the neural network, calculated from the loss/cost function
        // Error = Loss(Target, Output);
        // dY_dX means the derivative of Y in terms of X
        VectorF32 dError_dOutput{};
        for (int i = 1; i < nn.layer.size; i++)
        {
            dError_dOutput.size += nn.layer.data[i];
        }
        dError_dOutput.data = (float*)stack_alloc(&stack, dError_dOutput.size * sizeof(float));

        VectorF32 dOutput_dZ{};
        for (int i = 1; i < nn.layer.size; i++)
        {
            dOutput_dZ.size += nn.layer.data[i];
        }
        dOutput_dZ.data = (float*)stack_alloc(&stack, dOutput_dZ.size * sizeof(float));

        VectorF32 dError_dWeight{};
        dError_dWeight.size = nn.weight.size;
        dError_dWeight.data = (float*)stack_alloc(&stack, dError_dWeight.size * sizeof(float));

        VectorF32 dError_dBias{};
        dError_dBias.size = nn.bias.size;
        dError_dBias.data = (float*)stack_alloc(&stack, dError_dBias.size * sizeof(float));

        // expect neural network output / target output
        VectorF32 target{};
        target.size = nn.layer.data[nn.layer.size - 1];
        target.data = (float*)stack_alloc(&stack, target.size * sizeof(float));

        for (int epoch = 0; epoch < total_epochs; epoch++)
        {
            // shuffle the indices of training image data
            srand(time(NULL));
            for (int i = training_image_index.size - 1; i >= 0; i--)
            {
                int k = rand() % (i + 1);
                if (k == i) continue;
                int temp = training_image_index.data[k];
                training_image_index.data[k] = training_image_index.data[i];
                training_image_index.data[i] = temp;
            }

            // mini-batch
            for (int iteration = 0; iteration < training_per_epoch; iteration++)
            {
                memset(dError_dWeight.data, 0, dError_dWeight.size * sizeof(float));
                memset(dError_dBias.data, 0, dError_dBias.size * sizeof(float));

                const int training_example_start = iteration * mini_batch_size;
                for (int at_example = 0;
                    at_example < mini_batch_size && training_example_start + at_example < train_images_total;
                    at_example++)
                {
                    memset(dError_dOutput.data, 0, dError_dOutput.size * sizeof(float));
                    memset(dOutput_dZ.data, 0, dOutput_dZ.size * sizeof(float));

                    // the image we are training now
                    int image_index = training_image_index.data[training_example_start + at_example];
                    // the label correspond to the image we are training now
                    int label_index = image_index;

                    // target output
                    memset(target.data, 0, target.size * sizeof(float));
                    const uint8_t label = train_labels_data[label_index];
                    assert(label < target.size);
                    target.data[label] = 1.0f;

                    // forward pass / evaluation
                    {
                        // layer 0 / input layer init
                        assert(nn.layer.data[0] == train_image_size);
                        for (int pixel = 0; pixel < train_image_size; pixel++)
                        {
                            // Pixels are organized row-wise. 
                            // Pixel values are 0 to 255. 
                            // 0 means background (white), 255 means foreground (black).
                            // but for rgb, 0 is black, 255 is white
                            int data_index = image_index * train_image_size + pixel;
                            uint8_t value = train_images_data[data_index];
                            float normalization = value / 255.0f;
                            nn.neuron.data[pixel] = normalization;
                        }

                        // start from layer 1 / the first hidden layer
                        int neuron_data_offset = 0;
                        int weight_data_offset = 0;
                        int bias_data_offset = 0;
                        int dOutput_dZ_data_offset = 0;
                        for (int l = 1; l < nn.layer.size; l++)
                        {
                            const int prev_neuron_number = nn.layer.data[l - 1];
                            const int curr_neuron_number = nn.layer.data[l];

                            VectorF32 prev_neuron{};
                            prev_neuron.size = prev_neuron_number;
                            prev_neuron.data = nn.neuron.data + neuron_data_offset;

                            VectorF32 curr_neuron{};
                            curr_neuron.size = curr_neuron_number;
                            curr_neuron.data = nn.neuron.data + neuron_data_offset + prev_neuron_number;

                            MatrixF32 weight;
                            weight.data = nn.weight.data + weight_data_offset;
                            weight.row = curr_neuron_number;
                            weight.column = prev_neuron_number;

                            VectorF32 bias;
                            bias.data = nn.bias.data + bias_data_offset;
                            bias.size = curr_neuron_number;

                            // calculate neuron output value
                            for (int r = 0; r < curr_neuron_number; r++)
                            {
                                float Z = 0.0f;
                                const float b = bias.data[r];
                                for (int c = 0; c < prev_neuron_number; c++)
                                {
                                    int weight_data_index = r * prev_neuron_number + c;
                                    const float w = weight.data[weight_data_index];
                                    const float x = prev_neuron.data[c];
                                    Z += w * x;
                                }
                                Z += b;
                                // we are using sigmoid as activation function
                                float output = 1 / (1 + exp(-Z));
                                curr_neuron.data[r] = output;
                                // we also calculate dOutput/dZ
                                // we will use these value in backpropagation
                                float dO_dZ = output * (1.0f - output);
                                assert(dOutput_dZ_data_offset + r < dOutput_dZ.size);
                                dOutput_dZ.data[dOutput_dZ_data_offset + r] = dO_dZ;
                            }

                            const int weight_count = prev_neuron_number * curr_neuron_number;
                            const int bias_count = curr_neuron_number;
                            // update data offset
                            neuron_data_offset += prev_neuron_number;
                            weight_data_offset += weight_count;
                            bias_data_offset += bias_count;
                            dOutput_dZ_data_offset += curr_neuron_number;
                        }
                    }

                    // calculate error
                    // loss/cost function: 1/2 * (target - output)^2
                    {
                        float error = 0;
                        int output_layer_neuron_number = nn.layer.data[nn.layer.size - 1];
                        assert(output_layer_neuron_number == target.size);
                        for (int i = 0; i < output_layer_neuron_number; i++)
                        {
                            float t = target.data[i];
                            float y = nn.neuron.data[nn.neuron.size - output_layer_neuron_number + i];
                            error += (t - y) * (t - y) * 0.5f;
                            dError_dOutput.data[dError_dOutput.size - output_layer_neuron_number + i] = y - t;
                        }
                        fprintf(stdout, "epoch: %d, iteration: %d, image: %d, error: %f\n",
                            epoch, iteration, image_index, error);
                    }

                    // backward pass / backpropagation
                    {
                        int neuron_data_offset = nn.neuron.size;
                        int weight_data_offset = nn.weight.size;
                        int dE_dO_data_offset = dError_dOutput.size;
                        int dO_dZ_data_offset = dOutput_dZ.size;
                        int dE_dW_data_offset = dError_dWeight.size;
                        int dE_dB_data_offset = dError_dBias.size;
                        for (int l = nn.layer.size - 1; l > 0; l--)
                        {
                            const int prev_neuron_number = nn.layer.data[l - 1];
                            const int curr_neuron_number = nn.layer.data[l];

                            VectorF32 prev_neuron{};
                            prev_neuron.data = nn.neuron.data + neuron_data_offset - curr_neuron_number - prev_neuron_number;
                            prev_neuron.size = prev_neuron_number;

                            VectorF32 curr_neuron{};
                            curr_neuron.data = nn.neuron.data + neuron_data_offset - curr_neuron_number;
                            curr_neuron.size = curr_neuron_number;

                            VectorF32 dE_dO{};
                            dE_dO.data = dError_dOutput.data + dE_dO_data_offset - curr_neuron_number;
                            dE_dO.size = curr_neuron_number;

                            VectorF32 dO_dZ{};
                            dO_dZ.data = dOutput_dZ.data + dO_dZ_data_offset - curr_neuron_number;
                            dO_dZ.size = curr_neuron_number;

                            const int weight_count = prev_neuron_number * curr_neuron_number;
                            VectorF32 dE_dW{};
                            dE_dW.data = dError_dWeight.data + dE_dW_data_offset - weight_count;
                            dE_dW.size = weight_count;

                            const int bias_count = curr_neuron_number;
                            VectorF32 dE_dB{};
                            dE_dB.data = dError_dBias.data + dE_dB_data_offset - bias_count;
                            dE_dB.size = bias_count;

                            // calculate gradian
                            for (int i = 0; i < curr_neuron_number; i++)
                            {
                                const float _dE_dO = dE_dO.data[i];
                                const float _dO_dZ = dO_dZ.data[i];
                                for (int j = 0; j < prev_neuron_number; j++)
                                {
                                    const float _dZ_dW = prev_neuron.data[j];
                                    const float _dE_dW = _dZ_dW * _dO_dZ * _dE_dO;
                                    dE_dW.data[i * prev_neuron_number + j] += _dE_dW;
                                }
                                const float _dZ_dB = 1.0f;
                                const float _dE_dB = _dZ_dB * _dO_dZ * _dE_dO;
                                dE_dB.data[i] += _dE_dB;
                            }

                            VectorF32 weight{};
                            weight.data = nn.weight.data + weight_data_offset - weight_count;
                            weight.size = weight_count;

                            VectorF32 dE_dO_prev{};
                            dE_dO_prev.data = dError_dOutput.data + dE_dO_data_offset - curr_neuron_number - prev_neuron_number;
                            dE_dO_prev.size = prev_neuron_number;

                            // calculate dError/dOutput for previous layer neuron
                            // remeber we don't need calculate dError/dOutput for input layer
                            if (l - 1 > 0)
                            {
                                for (int i = 0; i < prev_neuron_number; i++)
                                {
                                    float _dE_dO_prev = 0;
                                    for (int j = 0; j < curr_neuron_number; j++)
                                    {
                                        const float _dE_dO = dE_dO.data[j];
                                        const float _dO_dZ = dO_dZ.data[j];
                                        const float _dZ_dO_prev = weight.data[j * prev_neuron_number + i];
                                        _dE_dO_prev += (_dZ_dO_prev * _dO_dZ * _dE_dO);
                                    }
                                    dE_dO_prev.data[i] = _dE_dO_prev;
                                }
                            }

                            // update data offset;
                            neuron_data_offset -= curr_neuron_number;
                            weight_data_offset -= weight_count;
                            dE_dO_data_offset -= curr_neuron_number;
                            dO_dZ_data_offset -= curr_neuron_number;
                            dE_dW_data_offset -= weight_count;
                            dE_dB_data_offset -= bias_count;
                        }
                    }
                }

                // calculate average gradian
                for (int i = 0; i < dError_dWeight.size; i++)
                {
                    dError_dWeight.data[i] = dError_dWeight.data[i] / mini_batch_size;
                }
                for (int i = 0; i < dError_dBias.size; i++)
                {
                    dError_dBias.data[i] = dError_dBias.data[i] / mini_batch_size;
                }

                // using gradian descent update weights and bias
                for (int i = 0; i < nn.weight.size; i++)
                {
                    float old_weight = nn.weight.data[i];
                    float new_weight = old_weight - (learning_rate * dError_dWeight.data[i]);
                    nn.weight.data[i] = new_weight;
                }
                for (int i = 0; i < nn.bias.size; i++)
                {
                    float old_bias = nn.bias.data[i];
                    float new_bias = old_bias - (learning_rate * dError_dBias.data[i]);
                    nn.bias.data[i] = new_bias;
                }
            }
        }

        stack_free(&stack, target.data);
        stack_free(&stack, dError_dBias.data);
        stack_free(&stack, dError_dWeight.data);
        stack_free(&stack, dOutput_dZ.data);
        stack_free(&stack, dError_dOutput.data);
        stack_free(&stack, training_image_index.data);
    }

    clock_t training_end = clock();
    float total_second = (float)(training_end - training_start) / CLOCKS_PER_SEC;
    fprintf(stdout, "Training completed. Total time spent: %f second\n", total_second);

    // test image data
    fp = fopen("mnist/t10k-images.idx3-ubyte", "rb");
    assert(fp != NULL);
    
    const int test_images_data_magic_number = mnist_read_int(fp);
    assert(test_images_data_magic_number == 2051);
    const int test_images_total = mnist_read_int(fp);
    const int test_image_pixel_rows = mnist_read_int(fp);
    assert(test_image_pixel_rows == train_image_pixel_rows);
    const int test_image_pixel_columns = mnist_read_int(fp);
    assert(test_image_pixel_columns == train_image_pixel_columns);

    const int test_image_size = test_image_pixel_rows * test_image_pixel_columns;
    assert(test_image_size == 784);
    const int test_images_data_size = test_image_size * test_images_total;
    const uint8_t* test_images_data =
        (uint8_t*)stack_alloc(&stack, test_images_data_size);
    assert(test_images_data != NULL);
    const size_t test_images_data_read_byte = 
        fread((void*)test_images_data, sizeof(uint8_t), test_images_data_size, fp);
    assert(test_images_data_read_byte == test_images_data_size);

    fclose(fp);
    
    // test label data
    fp = fopen("mnist/t10k-labels.idx1-ubyte", "rb");
    assert(fp != NULL);

    const int test_labels_magic_number = mnist_read_int(fp);
    assert(test_labels_magic_number == 2049);
    const int test_labels_total = mnist_read_int(fp);
    assert(test_labels_total == test_images_total);
    const uint8_t* test_labels_data =
        (uint8_t*)stack_alloc(&stack, test_labels_total * sizeof(uint8_t));
    assert(test_labels_data != NULL);

    const size_t test_labels_data_read_byte =
        fread((void*)test_labels_data, sizeof(uint8_t), test_labels_total, fp);
    assert(test_labels_data_read_byte == test_labels_total);

    fclose(fp);

    // test nerual network
	int fail_count = 0;
    {
        for (int i = 0; i < test_images_total; i++)
        {
            // the image we are testing now
            int image_index = i;
            // the label correspond to the image we are testing now
            int label_index = image_index;

            // expect output
            const uint8_t label = test_labels_data[label_index];

            // forward pass / evaluation
            {
                // layer 0 / input layer init
                assert(nn.layer.data[0] == test_image_size);
                for (int pixel = 0; pixel < test_image_size; pixel++)
                {
                    int data_index = image_index * test_image_size + pixel;
                    uint8_t value = test_images_data[data_index];
                    float normalization = value / 255.0f;
                    nn.neuron.data[pixel] = normalization;
                }

                // start from layer 1 / the first hidden layer
                int neuron_data_offset = 0;
                int weight_data_offset = 0;
                int bias_data_offset = 0;
                int dOutput_dZ_data_offset = 0;
                for (int l = 1; l < nn.layer.size; l++)
                {
                    const int prev_neuron_number = nn.layer.data[l - 1];
                    const int curr_neuron_number = nn.layer.data[l];

                    VectorF32 prev_neuron{};
                    prev_neuron.size = prev_neuron_number;
                    prev_neuron.data = nn.neuron.data + neuron_data_offset;

                    VectorF32 curr_neuron{};
                    curr_neuron.size = curr_neuron_number;
                    curr_neuron.data = nn.neuron.data + neuron_data_offset + prev_neuron_number;

                    MatrixF32 weight;
                    weight.data = nn.weight.data + weight_data_offset;
                    weight.row = curr_neuron_number;
                    weight.column = prev_neuron_number;

                    VectorF32 bias;
                    bias.data = nn.bias.data + bias_data_offset;
                    bias.size = curr_neuron_number;

                    // calculate neuron output value
                    for (int r = 0; r < curr_neuron_number; r++)
                    {
                        float Z = 0.0f;
                        float b = bias.data[r];
                        for (int c = 0; c < prev_neuron_number; c++)
                        {
                            int weight_data_index = r * prev_neuron_number + c;
                            float w = weight.data[weight_data_index];
                            float x = prev_neuron.data[c];
                            Z += w * x;
                        }
                        Z += b;
                        // we are using sigmoid as activation function
                        float output = 1 / (1 + exp(-Z));
                        curr_neuron.data[r] = output;
                    }

                    const int weight_count = prev_neuron_number * curr_neuron_number;
                    const int bias_count = curr_neuron_number;
                    // update data offset
                    neuron_data_offset += prev_neuron_number;
                    weight_data_offset += weight_count;
                    bias_data_offset += bias_count;
                    dOutput_dZ_data_offset += curr_neuron_number;
                }
            }

            // check output 
            {
                int result = 0;
                float max_neuron_output = FLT_MIN;
                int output_layer_neuron_number = nn.layer.data[nn.layer.size - 1];
                for (int i = 0; i < output_layer_neuron_number; i++)
                {
                    float y = nn.neuron.data[nn.neuron.size - output_layer_neuron_number + i];
                    if (y > max_neuron_output)
                    {
                        result = i;
                        max_neuron_output = y;
                    }
                }
                fprintf(stdout, "testing image: %d, expect output: %d, network output: %d, result: %s\n",
                    image_index, label, result, label == result ? "correct" : "wrong");
                if (label != result)
                {
                    fail_count++;
                }
            }
        }
        float accuracy = (float)(test_images_total - fail_count) / (float)test_images_total;
        fprintf(stdout, "Testing completed. Accuracy: %f%%\n", accuracy * 100);
    }

    return 0;
}