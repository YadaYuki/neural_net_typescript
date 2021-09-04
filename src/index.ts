import { TwoLayerNet } from './twoLayerNeuralnet';
import { loadMnist } from './data/load-mnist';
import nj from 'numjs';

const choice = (max: number, length: number) => {
  const randArray: number[] = new Array(length).fill(0);
  return randArray.map((_) => {
    return Math.floor(Math.random() * max);
  });
};

const range = (from: number, to: number, step = 1) => {
  const arr = [];
  for (let i = from; i < to; i += step) {
    arr.push(i);
  }
  return arr;
};

const getBatchData = (idxArr: number[], data: nj.NdArray<number[]>) => {
  const dataArr = data.tolist();
  return nj.array(
    idxArr.map((idx) => {
      return dataArr[idx];
    })
  );
};

const main = async () => {
  const { xTrain, yTrain } = await loadMnist();

  const network = new TwoLayerNet(784, 50, 10);
  const trainNum = xTrain.shape[0];
  const batchSize = 100;
  const iterNums = 1000;
  const learningRate = 0.1;
  let prev = Date.now();
  let now = Date.now();
  for (let i = 0; i < iterNums; i++) {
    const batchIdxList = choice(trainNum, batchSize);
    const xBatch = getBatchData(batchIdxList, xTrain);
    const yBatch = getBatchData(batchIdxList, yTrain);

    const loss = network.forward(xBatch, yBatch);
    network.backward();
    network.update(learningRate);
    if (i % 100 === 0) {
      now = Date.now();
      console.log(now - prev);
      console.log(loss);
      prev = now;
    }
  }
};
main();
