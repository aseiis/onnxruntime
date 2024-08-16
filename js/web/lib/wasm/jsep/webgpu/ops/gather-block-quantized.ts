// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { DataType } from '../../../wasm-common';
import { TensorView } from '../../tensor-view';
import { ShapeUtil } from '../../util';
import { AttributeWithCacheKey, createAttributeWithCacheKey } from '../attribute-with-cache-key';
import { ComputeContext, ProgramInfo, ProgramUniform } from '../types';

import {
  createTensorShapeVariables,
  inputVariable,
  outputVariable,
  ShaderHelper,
  tensorTypeToWsglValueType,
  UniformsArrayType,
} from './common';

export interface GatherBlockQuantizedAttributes extends AttributeWithCacheKey {
  gatherAxis: number;
  quantizeAxis: number;
  blockSize: number;
}

export const validateInputs = (inputs: readonly TensorView[], attributes: GatherBlockQuantizedAttributes): void => {
  if (inputs.length < 3 || inputs.length > 4) {
    throw new Error('GatherBlockQuantized requires 3 or 4 inputs.');
  }
  const quantizeAxis = ShapeUtil.normalizeAxis(attributes.quantizeAxis, inputs[0].dims.length);
  const blockSize = attributes.blockSize;
  const data = inputs[0];
  const scales = inputs[2];
  const zeroPoint = inputs.length === 4 ? inputs[3] : undefined;
  if (
    scales.dims.length !== data.dims.length ||
    !data.dims
      .map((d, i) => (i === quantizeAxis ? Math.ceil(d / blockSize) === scales.dims[i] : d === scales.dims[i]))
      .reduce((a, b) => a && b, true)
  ) {
    throw new Error(
      'Scales must have the same rank as the input tensor and the dims should match except on gatherAxis.',
    );
  }
  if (zeroPoint) {
    if (zeroPoint.dataType !== data.dataType) {
      throw new Error('Zero point must have the same data type as the input tensor.');
    }
    if (
      zeroPoint.dims.length !== scales.dims.length ||
      !zeroPoint.dims.map((d, i) => d === scales.dims[i]).reduce((a, b) => a && b, true)
    ) {
      throw new Error(
        'Zero point must have the same rank as the input tensor and the dims should match except on quantizeAxis.',
      );
    }
  }
};

const createGatherBlockQuantizedProgramInfo = (
  inputs: readonly TensorView[],
  attributes: GatherBlockQuantizedAttributes,
): ProgramInfo => {
  const inputShape = inputs[0].dims;
  const indicesShape = inputs[1].dims;
  const inputRank = inputShape.length;
  const gatherAxis = ShapeUtil.normalizeAxis(attributes.gatherAxis, inputRank);
  const quantizeAxis = ShapeUtil.normalizeAxis(attributes.quantizeAxis, inputRank);
  const outputShape = inputShape.slice(0);
  outputShape.splice(gatherAxis, 1, ...indicesShape);
  const outputSize = ShapeUtil.size(outputShape);
  const outputType = inputs[2].dataType;
  const inputType = inputs[0].dataType;
  const isSigned = inputType === DataType.int4; // input data type is either int4 or uint4.
  const programUniforms: ProgramUniform[] = [
    { type: DataType.uint32, data: outputSize },
    { type: DataType.uint32, data: quantizeAxis },
    { type: DataType.uint32, data: gatherAxis },
    { type: DataType.uint32, data: attributes.blockSize },
    ...createTensorShapeVariables(...inputs.map((input, _) => input.dims), outputShape),
  ];

  const getShaderSource = (shaderHelper: ShaderHelper) => {
    const data = inputVariable('data', inputs[0].dataType, inputs[0].dims.length);
    const indices = inputVariable('inputIndices', inputs[1].dataType, inputs[1].dims.length);
    const scales = inputVariable('scales', inputs[2].dataType, inputs[2].dims.length);
    const zeroPoint =
      inputs.length > 3 ? inputVariable('zeroPoint', inputs[3].dataType, inputs[3].dims.length) : undefined;
    const output = outputVariable('output', outputType, outputShape.length);
    const inputVariables = [data, indices, scales];
    if (zeroPoint) {
      inputVariables.push(zeroPoint);
    }
    const uniforms: UniformsArrayType = [
      { name: 'output_size', type: 'u32' },
      { name: 'quantize_axis', type: 'u32' },
      { name: 'gather_axis', type: 'u32' },
      { name: 'block_size', type: 'u32' },
    ];
    return `
        ${shaderHelper.registerUniforms(uniforms).declareVariables(...inputVariables, output)}
        ${shaderHelper.mainStart()}
        var output_indices = ${output.offsetToIndices('global_idx')};
        var indices_indices = ${indices.type.indices}(0);
        ${(() => {
          if (indicesShape.length > 1) {
            return `
          for (var i: u32 = 0; i < ${indicesShape.length}; i++) {
            var index = ${output.indicesGet('output_indices', 'uniforms.gather_axis + i')};
            ${indices.indicesSet('indices_indices', 'i', 'index')};
          }`;
          } else {
            return `indices_indices = ${output.indicesGet('output_indices', 'uniforms.gather_axis')};`;
          }
        })()};
        var data_indices = ${data.type.indices}(0);
        for (var i: u32 = 0; i < uniforms.gather_axis; i++) {
          data_indices[i] = output_indices[i];
        }
        data_indices[uniforms.gather_axis] = u32(${indices.getByIndices('indices_indices')});
        for (var i = uniforms.gather_axis + 1; i < ${outputShape.length}; i++) {
          data_indices[i] = output_indices[i + ${indicesShape.length} - 1];
        }
        var data_offset = ${data.indicesToOffset('data_indices')};
        var data_index = data_offset % 8;
        // Convert 4-bit packed data to 8-bit packed data.
        var packed_4bit_quantized_data = ${data.getByOffset('data_offset / 8')};
        var packed_8bit_quantized_data = (packed_4bit_quantized_data >> (4 * (data_index % 2))) & 0x0f0f0f0f;
        var quantized_data_vec = ${isSigned ? 'unpack4xI8' : 'unpack4xU8'}(u32(packed_8bit_quantized_data));
        var quantized_data = quantized_data_vec[data_index / 2];
        var scale_indices = data_indices;
        var quantize_axis_index = ${scales.indicesGet('data_indices', 'uniforms.quantize_axis')} / uniforms.block_size;
        ${scales.indicesSet('scale_indices', 'uniforms.quantize_axis', 'quantize_axis_index')};
        var scale = ${scales.getByIndices('scale_indices')};
        ${(() => {
          if (!zeroPoint) {
            return 'var zero_point = 0';
          } else {
            return `
              var zero_point_indices = scale_indices;
              var zero_point_offset = ${zeroPoint.indicesToOffset('zero_point_indices')};
              var zero_point_index = zero_point_offset % 8;
              var packed_4bit_zero_points = ${zeroPoint.getByOffset('zero_point_offset / 8')};
              var packed_8bit_zero_point = (packed_4bit_zero_points >> (4 * (zero_point_index % 2))) & 0x0f0f0f0f;
              var zero_point_vec = ${isSigned ? 'unpack4xI8' : 'unpack4xU8'}(u32(packed_8bit_zero_point));
              var zero_point = zero_point_vec[zero_point_index / 2];`;
          }
        })()};
        var dequantized_data = ${tensorTypeToWsglValueType(outputType)}(quantized_data - zero_point) * scale;
        ${output.setByOffset('global_idx', 'dequantized_data')};
    }`;
  };
  return {
    name: 'GatherBlockQuantized',
    shaderCache: {
      hint: `${attributes.cacheKey};${inputs.map((input, _) => input.dims.join('_')).join(';')}`,
      inputDependencies: Array.from({ length: inputs.length }, (_v, _i) => 'rank'),
    },
    getRunData: () => ({
      outputs: [{ dims: outputShape, dataType: outputType }],
      dispatchGroup: { x: Math.ceil(outputSize / 64 /* workgroup size */) },
      programUniforms,
    }),
    getShaderSource,
  };
};

export const gatherBlockQuantized = (context: ComputeContext, attributes: GatherBlockQuantizedAttributes): void => {
  const inputs = context.inputs;
  validateInputs(inputs, attributes);
  context.compute(createGatherBlockQuantizedProgramInfo(context.inputs, attributes));
};

export const parseGatherBlockQuantizedAttributes = (
  attributes: Record<string, unknown>,
): GatherBlockQuantizedAttributes =>
  createAttributeWithCacheKey({
    blockSize: attributes.blockSize as number,
    gatherAxis: attributes.gatherAxis as number,
    quantizeAxis: attributes.quantizeAxis as number,
  });