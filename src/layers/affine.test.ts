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
});
