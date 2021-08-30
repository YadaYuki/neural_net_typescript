// import { loadMnist } from './data/load-mnist';

/* TODO:implement MatrixType ... 全ての長さが同じ配列 */

export const sum = (a: number, b: number): number => {
  return a + b;
};

import nj from 'numjs';

console.log(nj.array([[0, -1, 3, 4, 5]]).T.tolist());
