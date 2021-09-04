import { TwoLayerNet } from './twoLayerNeuralnet';
import { loadMnist } from './data/load-mnist';
import { choice, getBatchData } from './utils';

const main = async () => {
  const { xTrain, yTrain, yTest, xTest } = await loadMnist();

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
  const testNum = yTest.shape[0];
  const testBatchIdxList = choice(testNum, 1000); // 1000個のデータで精度を検証する。
};

main();
