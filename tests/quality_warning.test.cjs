const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname, '../app/templates/index.html'), 'utf8');
const source = html.match(/function qualityWarning\(data\) \{[\s\S]*?\n\}/);
assert(source, 'quality warning function missing');
const context = vm.createContext({});
vm.runInContext(source[0], context);
const warning = context.qualityWarning;
const proved = {builds: [{}], search_status: {
    backend: 'max_quality_milp', state: 'FEASIBLE', optimality_proven: false,
    profile_count: 1, deep_optimization: true,
    models: [{state: 'FEASIBLE', priority_optimality_proven: true,
        cost_optimality_proven: false, validation_reason: 'passed', validation_failed: false}],
}};
const text = warning(proved);
assert.match(text, /Максимум основной цели подтверждён/);
assert.match(text, /Минимальная цена/);
assert.doesNotMatch(text, /Deep|профил/);
const multi = structuredClone(proved);
multi.search_status.profile_count = 2;
assert.doesNotMatch(warning(multi), /Максимум основной цели подтверждён/);
assert.match(warning(multi), /всем выбранным профилям/);
assert.doesNotMatch(warning(multi), /включить Deep/);
const changed = structuredClone(proved);
changed.roll_plan_search = {changed: true, keep_baseline: false};
assert.doesNotMatch(warning(changed), /Максимум основной цели подтверждён/);
const unproved = structuredClone(proved);
unproved.search_status.models[0].priority_optimality_proven = false;
unproved.search_status.deep_optimization = false;
assert.match(warning(unproved), /можно включить Deep/);
const old = structuredClone(proved);
delete old.search_status.profile_count;
assert.doesNotMatch(warning(old), /Максимум основной цели подтверждён/);
const invalid = structuredClone(proved);
invalid.search_status.models[0].validation_failed = true;
assert.doesNotMatch(warning(invalid), /Максимум основной цели подтверждён/);
const optimal = structuredClone(proved);
optimal.search_status.optimality_proven = true;
assert.equal(warning(optimal), '');
assert.equal(warning({builds: []}), '');
