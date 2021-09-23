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
    inputDim = { C: 1, Y: 28, X: 28 } as const,
    convParam = { filterNum: 30, filterSize: 5, pad: 0, stride: 1 } as const,
    poolingParam = { poolH: 2, poolW: 2, pad: 0, stride: 2 } as const,
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

  /* 基本的にはバッチ学習であることを前提とする。 */
  forward(
    xBatch: nj.NdArray<number[][][]>,
    tBatch: nj.NdArray<number[]>
  ): number {
    let scoreBatch: nj.NdArray<number[][][] | number[]> = xBatch;
    for (const layer of this.layers) {
      scoreBatch = layer.forwardBatch(scoreBatch);
    }
    const loss = this.lossLayer.forwardBatch(scoreBatch, tBatch);
    return loss;
  }
  backward(): void {
    let dout: nj.NdArray<number[] | number[][][]> =
      this.lossLayer.backwardBatch();
    const reversedLayers = this.layers.slice().reverse();
    for (const layer of reversedLayers) {
      dout = layer.backwardBatch(dout);
    }
  }
  update(learningRate = 0.1): void {
    const { dConvW, dConvB, dW1, db1, dW2, db2 } = this.gradient();
    this.convW = this.convW.subtract(dConvW.multiply(learningRate));
    this.convB = this.convB.subtract(dConvB.multiply(learningRate));
    this.W1 = this.W1.subtract(dW1.multiply(learningRate));
    this.b1 = this.b1.subtract(db1.multiply(learningRate));
    this.W2 = this.W2.subtract(dW2.multiply(learningRate));
    this.b2 = this.b2.subtract(db2.multiply(learningRate));
    (this.layers[0] as Convolution).W = this.convW;
    (this.layers[0] as Convolution).b = this.convB;
    (this.layers[3] as Affine).W = this.W1;
    (this.layers[3] as Affine).b = this.b1;
    (this.layers[5] as Affine).W = this.W2;
    (this.layers[5] as Affine).b = this.b2;
  }

  gradient(): {
    dConvW: nj.NdArray<number[][][]>;
    dConvB: nj.NdArray<number>;
    dW1: nj.NdArray<number[]>;
    db1: nj.NdArray<number>;
    dW2: nj.NdArray<number[]>;
    db2: nj.NdArray<number>;
  } {
    const conv = this.layers[0] as Convolution;
    const affine1 = this.layers[3] as Affine;
    const affine2 = this.layers[5] as Affine;
    return {
      dConvW: conv.dW,
      dConvB: conv.db,
      dW1: affine1.dW,
      db1: affine1.db,
      dW2: affine2.dW,
      db2: affine2.db,
    };
  }
}
