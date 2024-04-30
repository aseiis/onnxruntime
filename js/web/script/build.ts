// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as esbuild from 'esbuild';
import minimist from 'minimist';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import {SourceMapConsumer, SourceMapGenerator} from 'source-map';

console.time('BUILD');

/**
 * @summary Build script for ort-web using esbuild.
 */

const args = minimist(process.argv.slice(2));
/**
 * --bundle-mode=prod (default)
 *   Build multiple ort-web bundles for production.
 *
 * --bundle-mode=dev
 *   Build a single ort-web bundle for development, and a test bundle.
 *
 * --bundle-mode=perf
 *   Build a single ort-web bundle for performance test, and a test bundle.
 *
 * --bundle-mode=node
 *   Build a single ort-web bundle for nodejs.
 */
const BUNDLE_MODE: 'prod'|'dev'|'perf'|'node' = args['bundle-mode'] || 'prod';

/**
 * --debug
 *   Enable debug mode. In this mode, esbuild metafile feature will be enabled. Simple bundle analysis will be printed.
 *
 * --debug=verbose
 *   Enable debug mode. In this mode, esbuild metafile feature will be enabled. Detailed bundle analysis will be
 * printed.
 *
 * --debug=save
 *  Enable debug mode. In this mode, esbuild metafile feature will be enabled. Full bundle analysis will be saved to a
 * file as JSON.
 */
const DEBUG = args.debug;  // boolean|'verbose'|'save'

/**
 * Root folder of the source code: `<ORT_ROOT>/js/`
 */
const SOURCE_ROOT_FOLDER = path.join(__dirname, '../..');

/**
 * Default define values for the build.
 */
const DEFAULT_DEFINE = {
  'BUILD_DEFS.DISABLE_WEBGL': 'false',
  'BUILD_DEFS.DISABLE_JSEP': 'false',
  'BUILD_DEFS.DISABLE_WASM': 'false',
  'BUILD_DEFS.DISABLE_WASM_PROXY': 'false',
  'BUILD_DEFS.DISABLE_TRAINING': 'true',
  'BUILD_DEFS.PROXY_WORKER_URL': 'proxy.min.mjs',
  'BUILD_DEFS.ESM_IMPORT_META_URL': 'undefined',
};

const COPYRIGHT_HEADER = `/*!
 * ONNX Runtime Web v${require('../package.json').version}
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */`;

interface OrtBuildOptions {
  isProduction?: boolean;
  isNode?: boolean;
  format: 'iife'|'cjs'|'esm';
  outputBundleName: string;
  define?: Record<string, string>;
}

const alreadyBuilt = new Set();

async function buildBundle(options: esbuild.BuildOptions) {
  // Skip if the same build options have been built before.
  const serializedOptions = JSON.stringify(options);
  if (alreadyBuilt.has(serializedOptions)) {
    return;
  } else {
    alreadyBuilt.add(serializedOptions);
  }

  // Patch banner:
  //
  // - Add copy right header.
  // - For Node + ESM, add a single line fix to make it work.
  //   (see: https://github.com/evanw/esbuild/pull/2067#issuecomment-1981642558)
  const NODE_ESM_FIX_MIN = 'import{createRequire}from"module";const require=createRequire(import.meta.url);';
  const banner = {
    js: options.platform === 'node' && options.format === 'esm' ? COPYRIGHT_HEADER + '\n' + NODE_ESM_FIX_MIN :
                                                                  COPYRIGHT_HEADER
  };

  // Patch footer:
  //
  // For IIFE format, add a custom footer to make it compatible with CommonJS module system.
  // For other formats, no footer is needed.
  //
  // ESBuild does not support UMD format (which is a combination of IIFE and CommonJS). We don't want to generate 2
  // build targets (IIFE and CommonJS) because it will increase the package size. Instead, we generate IIFE and append
  // this footer to make it compatible with CommonJS module system.
  //
  // see also: https://github.com/evanw/esbuild/issues/507
  //
  const COMMONJS_FOOTER_MIN = 'typeof exports=="object"&&typeof module=="object"&&(module.exports=ort);';
  const footer = options.format === 'iife' ? {js: COMMONJS_FOOTER_MIN} : undefined;

  // For esm, set BUILD_DEFS.ESM_IMPORT_META_URL to the current file's URL.
  if (options.format === 'esm') {
    options.define = {...options.define, 'BUILD_DEFS.ESM_IMPORT_META_URL': 'import.meta.url'};
  }

  const result = await esbuild.build({
    logLevel: DEBUG ? (DEBUG === 'verbose' || DEBUG === 'save' ? 'verbose' : 'debug') : 'info',
    metafile: !!DEBUG,
    absWorkingDir: SOURCE_ROOT_FOLDER,
    bundle: true,
    banner,
    footer,
    ...options
  });
  if (DEBUG) {
    if (DEBUG === 'save') {
      await fs.writeFile(
          `${path.basename(options.outfile!)}.esbuild.metafile.json`, JSON.stringify(result.metafile!, null, 2));
    } else {
      console.log(await esbuild.analyzeMetafile(result.metafile!, {verbose: DEBUG === 'verbose'}));
    }
  }
}

async function buildOrt({
  isProduction = false,
  isNode = false,
  format,
  outputBundleName,
  define = DEFAULT_DEFINE,
}: OrtBuildOptions) {
  // distribution code is split into multiple files:
  // - [bundle-name][.min].[m]js
  // - [bundle-name].proxy[.min].mjs
  // - ort[-training]-wasm[-simd][-threaded][.jsep].mjs
  const platform = isNode ? 'node' : 'browser';
  const external = isNode ? ['onnxruntime-common'] : ['node:fs/promises', 'node:fs', 'node:os'];
  const plugins: esbuild.Plugin[] = [];

  // Build proxy worker bundle if needed.
  if (define['BUILD_DEFS.DISABLE_WASM_PROXY'] !== 'true') {
    const fileName = `${
        outputBundleName.endsWith('.min') ? outputBundleName.substring(0, outputBundleName.length - 4) :
                                            outputBundleName}.proxy${isProduction ? '.min' : ''}.mjs`;
    define['BUILD_DEFS.PROXY_WORKER_URL'] = JSON.stringify(fileName);
    await buildBundle({
      entryPoints: ['web/lib/wasm/proxy-worker/main.ts'],
      outfile: `web/dist/${fileName}`,
      platform,
      format: 'esm',
      plugins,
      external,
      define: {
        ...define,
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
      },
      sourcemap: isProduction ? 'external' : 'inline',
      minify: isProduction,
    });
  }

  await buildBundle({
    entryPoints: ['web/lib/index.ts'],
    outfile: `web/dist/${outputBundleName}.${format === 'esm' ? 'mjs' : 'js'}`,
    platform,
    format,
    globalName: 'ort',
    plugins,
    external,
    define,
    sourcemap: isProduction ? 'linked' : 'inline',
    minify: isProduction,
  });
}

async function buildTest() {
  const isProduction = BUNDLE_MODE === 'perf';

  await buildBundle({
    absWorkingDir: path.join(SOURCE_ROOT_FOLDER, 'web/test'),

    entryPoints: ['test-main.ts'],
    outfile: isProduction ? 'ort.test.min.js' : 'ort.test.js',
    platform: 'browser',
    format: 'iife',
    define: DEFAULT_DEFINE,
    sourcemap: isProduction ? false : 'inline',
    sourceRoot: path.join(SOURCE_ROOT_FOLDER, 'web/test'),
    external: ['../../node'],
    plugins: [
      // polyfill nodejs modules
      require('esbuild-plugin-polyfill-node').polyfillNode({globals: false}),
      // make "ort" external
      {
        name: 'make-ort-external',
        setup(build: esbuild.PluginBuild) {
          build.onResolve(
              {filter: /^onnxruntime-common$/},
              _args => ({path: 'onnxruntime-common', namespace: 'make-ort-external'}));
          build.onLoad(
              {filter: /.*/, namespace: 'make-ort-external'},
              _args => ({contents: 'module.exports = globalThis.ort;'}));
        }
      }
    ],
    minify: isProduction,
  });
}


/**
 * Perform the post-process step after ESBuild finishes the build.
 *
 * This is a custom post process step to insert magic comments to a specific import call:
 * ```
 * ... await import(...
 * ```
 * to:
 * ```
 * ... await import(/* webpackIgnore: true *\/...
 * ```
 *
 * Why we need this?
 *
 * If a project uses Webpack to bundle the code, Webpack will try to resolve the dynamic import calls. However, we don't
 * want Webpack to resolve the dynamic import calls inside the ort-web bundle because:
 *
 * - We want to keep the ort-*.mjs and ort-*.wasm as-is. This makes it able to replace the ort-*.mjs and ort-*.wasm with
 * a custom build if needed.
 * - The Emscripten generated code uses `require()` to load Node.js modules. Those code is guarded by a feature check to
 * make sure only run in Node.js. Webpack does not recognize the feature check and will try to resolve the `require()`
 * in browser environment. This will cause the Webpack build to fail.
 * - There are multiple entry points that use dynamic import to load the ort-*.mjs and ort-*.wasm. If the content of the
 * dynamic import is resolved by Webpack, it will be duplicated in the final bundle. This will increase the bundle size.
 *
 * What about other bundlers?
 *
 * TBD
 *
 */
async function postProcess() {
  const IMPORT_MAGIC_COMMENT = '/*webpackIgnore:true*/';
  const IMPORT_ORIGINAL = 'await import(';
  const IMPORT_NEW = `await import(${IMPORT_MAGIC_COMMENT}`;

  const files = await fs.readdir(path.join(SOURCE_ROOT_FOLDER, 'web/dist'));
  for (const file of files) {
    // only process on "ort.*.min.js" and "ort.*.min.mjs" files.
    if ((file.endsWith('.min.js') || file.endsWith('.min.mjs')) && file.startsWith('ort.')) {
      const jsFilePath = path.join(SOURCE_ROOT_FOLDER, 'web/dist', file);
      const sourcemapFilePath = jsFilePath + '.map';

      const originalJsFileSize = (await fs.stat(jsFilePath)).size;

      if (!files.includes(file + '.map')) {
        continue;
      }

      const jsFileLines = (await fs.readFile(jsFilePath, 'utf-8')).split('\n');

      let line = -1, column = -1, found = false;
      for (let i = 0; i < jsFileLines.length; i++) {
        const importColumnIndex = jsFileLines[i].indexOf(IMPORT_ORIGINAL);
        if (importColumnIndex !== -1) {
          if (found || importColumnIndex !== jsFileLines[i].lastIndexOf(IMPORT_ORIGINAL)) {
            throw new Error(`Multiple dynamic import calls found in "${jsFilePath}". Should not happen.`);
          }
          line = i + 1;
          column = importColumnIndex + IMPORT_ORIGINAL.length;
          jsFileLines[i] = jsFileLines[i].replace(IMPORT_ORIGINAL, IMPORT_NEW);
          found = true;
        }
      }
      if (!found) {
        if (file.includes('webgl')) {
          // skip webgl
          continue;
        }
        throw new Error(`Dynamic import call not found in "${jsFilePath}". Should not happen.`);
      }

      const originalSourcemapString = await fs.readFile(sourcemapFilePath, 'utf-8');
      await SourceMapConsumer.with(originalSourcemapString, null, async (consumer) => {
        // create new source map and set source content
        const updatedSourceMap = new SourceMapGenerator();
        for (const source of consumer.sources) {
          const content = consumer.sourceContentFor(source);
          if (!content) {
            throw new Error(`Source content not found for source "${source}".`);
          }
          updatedSourceMap.setSourceContent(source, content);
        }

        consumer.eachMapping((mapping) => {
          if (mapping.generatedLine === line && mapping.generatedColumn >= column) {
            mapping.generatedColumn += IMPORT_MAGIC_COMMENT.length;
          }

          updatedSourceMap.addMapping({
            generated: {line: mapping.generatedLine, column: mapping.generatedColumn},
            source: mapping.source,
            original: {line: mapping.originalLine, column: mapping.originalColumn},
            name: mapping.name,
          });
        });

        const updatedSourcemapString = updatedSourceMap.toString();

        // perform simple validation
        const originalSourcemap = JSON.parse(originalSourcemapString);
        const updatedSourcemap = JSON.parse(updatedSourcemapString);

        if (originalSourcemap.sources.length !== updatedSourcemap.sources.length ||
            originalSourcemap.sourcesContent.length !== updatedSourcemap.sourcesContent.length ||
            new Set(originalSourcemap.names).size !== new Set(updatedSourcemap.names).size) {
          throw new Error('Failed to update source map: source map length mismatch.');
        }
        const originalMappingsCount = originalSourcemap.mappings.split(/[;,]/);
        const updatedMappingsCount = updatedSourcemap.mappings.split(/[;,]/);
        if (originalMappingsCount.length !== updatedMappingsCount.length) {
          throw new Error('Failed to update source map: mappings count mismatch.');
        }

        await fs.writeFile(sourcemapFilePath, updatedSourcemapString);
      });

      await fs.writeFile(jsFilePath, jsFileLines.join('\n'));
      const newJsFileSize = (await fs.stat(jsFilePath)).size;
      if (newJsFileSize - originalJsFileSize !== IMPORT_MAGIC_COMMENT.length) {
        throw new Error(`Failed to insert magic comment to file "${file}". Original size: ${
            originalJsFileSize}, New size: ${newJsFileSize}`);
      }
    }
  }
}

async function validate() {
  const files = await fs.readdir(path.join(SOURCE_ROOT_FOLDER, 'web/dist'));
  for (const file of files) {
    // validate on all "ort.*.min.js" and "ort.*.min.mjs" files.
    if ((file.endsWith('.js') || file.endsWith('.mjs')) && file.startsWith('ort.')) {
      const isMinified = file.endsWith('.min.js') || file.endsWith('.min.mjs');
      const content = await fs.readFile(path.join(SOURCE_ROOT_FOLDER, 'web/dist', file), 'utf-8');

      if (isMinified) {
        // all files should not contain BUILD_DEFS definition. BUILD_DEFS should be defined in the build script only.
        //
        // If the final bundle contains BUILD_DEFS definition, it means the build script is not working correctly. In
        // this case, we should fix the build script (this file).
        //
        if (content.includes('BUILD_DEFS')) {
          throw new Error(`Validation failed: "${file}" contains BUILD_DEFS definition.`);
        }
      }

      // all files should contain the magic comment to ignore dynamic import calls.
      //
      if (!file.includes('webgl')) {
        const contentToSearch = isMinified ? '/*webpackIgnore:true*/' : '/* webpackIgnore: true */';
        if (!content.includes(contentToSearch)) {
          throw new Error(`Validation failed: "${file}" does not contain magic comment.`);
        }
      }
    }
  }
}

async function main() {
  console.timeLog('BUILD', 'Start building ort-web bundles...');

  /**
   * add all 4 build tasks for web bundles. Includes:
   * - IIFE/CJS, debug:                [name].js
   * - IIFE/CJS, production:           [name].min.js
   * - ESM, debug:                     [name].mjs
   * - ESM, production:                [name].min.mjs
   */
  const addAllWebBuildTasks = async (options: Omit<OrtBuildOptions, 'format'>) => {
    // [name].js
    await buildOrt({
      ...options,
      format: 'iife',
    });
    // [name].min.js
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName + '.min',
      format: 'iife',
      isProduction: true,
    });
    // [name].mjs
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName,
      format: 'esm',
    });
    // [name].min.mjs
    await buildOrt({
      ...options,
      outputBundleName: options.outputBundleName + '.min',
      format: 'esm',
      isProduction: true,
    });
  };

  if (BUNDLE_MODE === 'node' || BUNDLE_MODE === 'prod') {
    // ort.node.min.js
    await buildOrt({
      isProduction: true,
      isNode: true,
      format: 'cjs',
      outputBundleName: 'ort.node.min',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
      },
    });
    // ort.node.min.mjs
    await buildOrt({
      isProduction: true,
      isNode: true,
      format: 'esm',
      outputBundleName: 'ort.node.min',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
      },
    });
  }

  if (BUNDLE_MODE === 'dev') {
    // ort.all.js
    await buildOrt({outputBundleName: 'ort.all', format: 'iife', define: {...DEFAULT_DEFINE}});
  }

  if (BUNDLE_MODE === 'perf') {
    // ort.all.min.js
    await buildOrt({
      isProduction: true,
      outputBundleName: 'ort.all.min',
      format: 'iife',
    });
  }

  if (BUNDLE_MODE === 'prod') {
    // ort.all[.min].[m]js
    await addAllWebBuildTasks({outputBundleName: 'ort.all'});

    // ort[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_JSEP': 'true'},
    });
    // ort.webgpu[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgpu',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_WEBGL': 'true'},
    });
    // ort.wasm[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm',
      define: {...DEFAULT_DEFINE, 'BUILD_DEFS.DISABLE_JSEP': 'true', 'BUILD_DEFS.DISABLE_WEBGL': 'true'},
    });
    // ort.webgl[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.webgl',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WASM': 'true',
      },
    });
    // ort.wasm-core[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.wasm-core',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
        'BUILD_DEFS.DISABLE_WASM_PROXY': 'true',
      },
    });
    // ort.training.wasm[.min].[m]js
    await addAllWebBuildTasks({
      outputBundleName: 'ort.training.wasm',
      define: {
        ...DEFAULT_DEFINE,
        'BUILD_DEFS.DISABLE_TRAINING': 'false',
        'BUILD_DEFS.DISABLE_JSEP': 'true',
        'BUILD_DEFS.DISABLE_WEBGL': 'true',
      },
    });
  }

  if (BUNDLE_MODE === 'dev' || BUNDLE_MODE === 'perf') {
    await buildTest();
  }

  if (BUNDLE_MODE === 'prod') {
    console.timeLog('BUILD', 'Start post-processing...');
    await postProcess();

    console.timeLog('BUILD', 'Start validating...');
    await validate();
  }

  console.timeEnd('BUILD');
}

void main();
