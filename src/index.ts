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
  // for (let i = 0; i < iterNums; i++) {
  // const batchIdxList = choice(trainNum, batchSize);
  const batchIdxList = range(0, 100);
  console.log(batchIdxList);
  const xBatch = getBatchData(batchIdxList, xTrain);
  const yBatch = getBatchData(batchIdxList, yTrain);

  const loss = network.forward(xBatch, yBatch);
  network.backward();
  const { dW1, db1, dW2, db2 } = network.gradient();
  network.W1 = network.W1.subtract(dW1.multiply(learningRate));
  network.b1 = network.b1.subtract(db1.multiply(learningRate));
  network.W2 = network.W2.subtract(dW2.multiply(learningRate));
  network.b2 = network.b2.subtract(db2.multiply(learningRate));
  // if (i % 100 === 0) {
  now = Date.now();
  console.log(now - prev);
  console.log(network.W1.get(0, 0));
  console.log(network.W2.get(0, 0));
  console.log(loss);
  prev = now;
  // }
  // }
};
main();

// const a = nj.array([1, 2, 3, 4, 5, 6]);
// console.log(a);
// console.log(a.subtract([1, 1, 1, 1, 1, 1]));
// console.log(a);
