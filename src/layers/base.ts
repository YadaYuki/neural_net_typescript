/*

*/
export interface Layer {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  forward(...arg: any): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  backward(...arg: any): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  forwardBatch(...arg: any): any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  backwardBatch(...arg: any): any;
}
