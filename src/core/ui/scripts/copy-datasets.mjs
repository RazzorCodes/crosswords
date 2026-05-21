import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

const repoRoot = resolve(import.meta.dirname, '../../../..');
const outputRoot = resolve(import.meta.dirname, '../public/datasets');
const sources = [
  { prefix: 'regular', group: 'regular', sourceDir: join(repoRoot, 'data/regular') },
  { prefix: 'hq', group: 'hq', sourceDir: join(repoRoot, 'data/high_quality') },
];
const userInputSources = new Set(['keyboard-correction']);

function readJsonl(path) {
  return readFileSync(path, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function writeJsonl(path, records) {
  writeFileSync(path, `${records.map((record) => JSON.stringify(record)).join('\n')}\n`);
}

rmSync(outputRoot, { recursive: true, force: true });
mkdirSync(outputRoot, { recursive: true });

const datasets = [];
for (const source of sources) {
  const files = readdirSync(source.sourceDir).filter((file) => file.endsWith('.jsonl')).sort();
  for (const file of files) {
    const inputPath = join(source.sourceDir, file);
    const records = readJsonl(inputPath).filter((record) => !record.tombstoned);
    const outputName = source.group === 'hq' ? file.replace(/^high_quality-/, 'hq-') : file;
    writeJsonl(join(outputRoot, outputName), records);
    datasets.push({
      id: outputName.replace(/\.jsonl$/, ''),
      group: source.group,
      label: outputName.replace(/\.jsonl$/, ''),
      url: `datasets/${outputName}`,
      count: records.length,
    });

    const inputted = records.filter((record) => userInputSources.has(record.source));
    if (inputted.length > 0) {
      const inputtedName = `inputted-${outputName}`;
      writeJsonl(join(outputRoot, inputtedName), inputted);
      datasets.push({
        id: inputtedName.replace(/\.jsonl$/, ''),
        group: 'inputted',
        label: inputtedName.replace(/\.jsonl$/, ''),
        url: `datasets/${inputtedName}`,
        count: inputted.length,
      });
    }
  }
}

writeFileSync(
  join(outputRoot, 'manifest.json'),
  `${JSON.stringify({ version: new Date().toISOString(), datasets }, null, 2)}\n`,
);

console.log(`Copied ${datasets.length} handwriting lab datasets to ${basename(outputRoot)}.`);
