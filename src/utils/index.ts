import nj from 'numjs';

export const choice = (max: number, length: number): number[] => {
  const randArray: number[] = new Array(length).fill(0);
  return randArray.map((_) => {
    return Math.floor(Math.random() * max);
  });
};

export const range = (from: number, to: number, step = 1): number[] => {
  const arr = [];
  for (let i = from; i < to; i += step) {
    arr.push(i);
  }
  return arr;
};

export const getBatchData = (
  idxArr: number[],
  data: nj.NdArray<number[]>
): nj.NdArray<number[]> => {
  const dataArr = data.tolist();
  return nj.array(
    idxArr.map((idx) => {
      return dataArr[idx];
    })
  );
};

export const maxIdx = (arr: number[]): number => {
  return arr.reduce((acc, cur, idx, src) => (src[acc] < cur ? idx : acc));
};
