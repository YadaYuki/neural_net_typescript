import nj from 'numjs';
import { Layer } from '../layers/base';
import { Affine } from '../layers/affine';
import { Relu } from '../layers/relu';
import { ImageRelu } from '../layers/imageRelu';
import { SoftmaxWithLoss } from '../layers/softmaxWithLoss';
import { Convolution } from '../layers/convolution';
import { Pooling } from '../layers/pooling';

export class SimpleConvNet {
  convW: nj.NdArray<number[][][]>;
  convB: nj.NdArray<number>;
  W1: nj.NdArray<number[]>;
  b1: nj.NdArray<number>;
  W2: nj.NdArray<number[]>;
  b2: nj.NdArray<number>;
  layers: Layer[];
  lossLayer: Layer;
  constructor(
    inputDim = { C: 1, Y: 28, X: 28 },
    convParam = { filterNum: 30, filterSize: 5, pad: 0, stride: 1 },
    poolingParam = { poolH: 2, poolW: 2, pad: 0, stride: 1 },
    hiddenSize = 100,
    outputSize = 10,
    weightInitStd = 0.01
  ) {
    const convOutputSize =
      (inputDim.Y - convParam.filterSize + 2 * convParam.pad) /
        convParam.stride +
      1;
    const poolOutputSize = Math.floor(
      convParam.filterNum * (convOutputSize / 2) * (convOutputSize / 2)
    );

    this.convW = nj
      .random([
        convParam.filterNum,
        inputDim.C,
        convParam.filterSize,
        convParam.filterSize,
      ])
      .multiply(weightInitStd)
      .reshape(
        convParam.filterNum,
        inputDim.C,
        convParam.filterSize,
        convParam.filterSize
      );
    this.convB = nj.random(convParam.filterNum).multiply(weightInitStd);
    this.W1 = nj
      .random([poolOutputSize * hiddenSize])
      .multiply(weightInitStd)
      .reshape(poolOutputSize, hiddenSize) as nj.NdArray<number[]>;
    this.b1 = nj.zeros([hiddenSize]);
    this.W2 = nj
      .random([hiddenSize * outputSize])
      .multiply(weightInitStd)
      .reshape(hiddenSize, outputSize) as nj.NdArray<number[]>;
    this.b2 = nj.zeros([outputSize]);

    this.layers = [
      new Convolution(this.convW, this.convB, convParam.stride, convParam.pad),
      new ImageRelu(),
      new Pooling(
        poolingParam.poolH,
        poolingParam.poolW,
        poolingParam.stride,
        poolingParam.pad
      ),
      new Affine(this.W1, this.b1),
      new Relu(),
      new Affine(this.W2, this.b2),
    ];
    this.lossLayer = new SoftmaxWithLoss();
  }
}
