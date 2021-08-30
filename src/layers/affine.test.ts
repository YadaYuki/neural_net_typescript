import { Affine } from './affine';
import nj from 'numjs';

describe('Affine Layer Test', () => {
  describe('Affine.forward', () => {
    test('forward x=[1,1],W=[[3,3,3],[3,3,3]],b=[0,0,0] should return [6,6,6]', () => {
      const affine = new Affine(
        nj.array([
          [3, 3, 3],
          [3, 3, 3],
        ]),
        nj.array([0, 0, 0])
      );
      const x = nj.array([1, 1]);
      expect(affine.forward(x).tolist()).toEqual([6, 6, 6]);
    });
    test('forward x=[1,1] W = [[3,3,3,],[3,3,3,]],b=[1,1,1] should [7,7,7]', () => {
      const affine = new Affine(
        nj.array([
          [3, 3, 3],
          [3, 3, 3],
        ]),
        nj.array([1, 1, 1])
      );
      const x = nj.array([1, 1]);
      expect(affine.forward(x).tolist()).toEqual([7, 7, 7]);
    });
    test('forward x=[5,2,1] W = [[3,3,3,],[3,3,3,],[3,3,3,]],b=[1,1,1] should [25,25,25]', () => {
      const affine = new Affine(
        nj.array([
          [3, 3, 3],
          [3, 3, 3],
          [3, 3, 3],
        ]),
        nj.array([1, 1, 1])
      );
      const x = nj.array([5, 2, 1]);
      expect(affine.forward(x).tolist()).toEqual([25, 25, 25]);
    });
  });
  describe('Affine.forwardBatch', () => {
    test('forwardBatch x=[[5,2,1],[5,2,1],[5,2,1]] W = [[3,3,3,],[3,3,3,],[3,3,3,]],b=[1,1,1] should [[25,25,25],[25,25,25],[25,25,25]]', () => {
      const affine = new Affine(
        nj.array([
          [3, 3, 3],
          [3, 3, 3],
          [3, 3, 3],
        ]),
        nj.array([1, 1, 1])
      );
      const x = nj.array([
        [5, 2, 1],
        [5, 2, 1],
        [5, 2, 1],
      ]);
      expect(affine.forwardBatch(x).tolist()).toEqual([
        [25, 25, 25],
        [25, 25, 25],
        [25, 25, 25],
      ]);
    });
  });
  describe('Affine.backward', () => {
    test('backward ', () => {
      const affine = new Affine(
        nj.array([
          [3, 3, 3],
          [3, 3, 3],
          [3, 3, 3],
        ]),
        nj.array([1, 1, 1]),
        nj.array([1, 2, 3])
      );
      const dx = affine.backward(nj.array([3, 3, 3]));
      expect(affine.db.tolist()).toEqual([3, 3, 3]);
      expect(affine.dW.tolist()).toEqual([
        [3, 3, 3],
        [6, 6, 6],
        [9, 9, 9],
      ]);
      expect(dx.tolist()).toEqual([27, 27, 27]);
    });
  });
  describe('Affine.backwardBatch', () => {
    test('backwardBatch ', () => {
      const affine = new Affine(
        nj.array([
          [1, 3, 3],
          [3, 5, 3],
          [1, 6, 9],
        ]),
        nj.array([1, 1, 1]),
        undefined,
        nj.array([
          [1, 1, 1],
          [2, 3, 1],
          [4, 2, 1],
        ])
      );
      const dx = affine.backwardBatch(
        nj.array([
          [1, 4, 3],
          [4, 2, 6],
          [3, 2, 5],
        ])
      );
      expect(affine.db.tolist()).toEqual([8, 8, 14]);
      expect(affine.dW.tolist()).toEqual([
        [21, 16, 35],
        [19, 14, 31],
        [8, 8, 14],
      ]);
      expect(dx.tolist()).toEqual([
        [22, 32, 52],
        [28, 40, 70],
        [24, 34, 60],
      ]);
    });
  });
});
