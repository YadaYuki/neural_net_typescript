// import { loadMnist } from './data/load-mnist';

// loadMnist();

export const sum = (a: number, b: number): number => {
  return a + b;
};

import nj from 'numjs';

console.log(nj.array([1, 1, 1]).reshape(1, 3).tolist());
