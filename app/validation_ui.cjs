// Real headless Chrome / CDP, no npm dependencies or external HTTP requests.
const fs = require('node:fs');
const [wsUrl, pageUrl, output] = process.argv.slice(2);
const ws = new WebSocket(wsUrl);
const pending = new Map();
const requests = [];
const errors = [];
let serial = 0;
ws.onmessage = event => {
    const m = JSON.parse(event.data);
    if (m.id && pending.has(m.id)) {
        const {resolve, reject} = pending.get(m.id); pending.delete(m.id);
        if (m.error) reject(new Error(JSON.stringify(m.error))); else resolve(m.result);
    }
    if (m.method === 'Network.requestWillBeSent' && m.params.request.url.includes('/api/'))
        requests.push(m.params.request.url);
    if (m.method === 'Runtime.exceptionThrown') errors.push(m.params.exceptionDetails);
};
function command(method, params={}) {
    return new Promise((resolve, reject) => {
        const id = ++serial; pending.set(id, {resolve, reject});
        ws.send(JSON.stringify({id, method, params}));
    });
}
async function evaluate(expression) {
    const r = await command('Runtime.evaluate', {expression, returnByValue: true, awaitPromise: true});
    if (r.exceptionDetails) throw new Error(JSON.stringify(r.exceptionDetails));
    return r.result.value;
}
async function wait(expression) {
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
        if (await evaluate(expression)) return;
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    throw new Error('Timeout: ' + expression);
}
function assert(condition, message) { if (!condition) throw new Error(message); }
function last(path) {
    const r = requests.filter(url => new URL(url).pathname === path).at(-1);
    assert(r, 'Missing request ' + path); return new URL(r).searchParams;
}
async function clickBuild() {
    const count = requests.filter(url => new URL(url).pathname === '/api/builds').length;
    await evaluate("document.getElementById('buildBtn').click(); findBuilds()");
    await wait("!document.getElementById('buildBtn').disabled");
    assert(requests.filter(url => new URL(url).pathname === '/api/builds').length === count + 1, 'Build request missing');
    assert(await evaluate("document.getElementById('results').textContent.includes('Найдено')"),
           'No successful rendered builds: ' + await evaluate("document.getElementById('results').textContent"));
    return last('/api/builds');
}
(async () => {
    await new Promise((resolve, reject) => {ws.onopen = resolve; ws.onerror = reject;});
    await command('Page.enable'); await command('Runtime.enable'); await command('Network.enable');
    await command('Emulation.setDeviceMetricsOverride', {width:1280,height:900,deviceScaleFactor:1,mobile:false});
    await command('Network.setBlockedURLs', {urls: ['https://*', 'http://*.com/*', 'http://*.ru/*']});
    await command('Page.navigate', {url: pageUrl});
    await wait("document.getElementById('prio_sprint_speed_modifier') && document.getElementById('armorId').options.length > 1");
    assert(await evaluate("document.getElementById('prio_radiation_accumulation').parentElement.textContent.includes('(защита)')"), 'Danger direction label');
    assert(await evaluate("!document.getElementById('prio_bleeding_accumulation').parentElement.textContent.includes('(защита)')"), 'Bleeding must not claim the opposite objective direction');
    await evaluate(`
        document.getElementById('budget').value='11880000';
        document.getElementById('tolerance').value='0';
        document.getElementById('containerId').value='dkln';
        document.getElementById('allowDuplicates').value='1';
        document.getElementById('topN').value='5';
        document.getElementById('armorId').value='3g6p5';
        document.getElementById('armorId').dispatchEvent(new Event('change'));
        document.getElementById('prio_sprint_speed_modifier').value='10';
        document.getElementById('prio_health_bonus').value='8';
        document.getElementById('prio_health_bonus').dispatchEvent(new Event('input',{bubbles:true}));
        document.getElementById('min_health_bonus').value='0';
        document.getElementById('max_radiation_accumulation').value='5';
    `);
    const checks = [];
    let p = await clickBuild();
    assert(p.get('optimization_mode') === 'balanced', 'Default goal');
    assert(p.get('budget') === '11880000' && p.get('tolerance') === '0', 'Budget');
    assert(p.get('armor_id') === '3g6p5', 'Armor');
    assert(JSON.parse(p.get('priorities')).health_bonus === 8, 'Health weight');
    assert(JSON.parse(p.get('min_thresholds')).health_bonus === 0, 'Minimum');
    assert(JSON.parse(p.get('max_thresholds')).radiation_accumulation === 5, 'Maximum');
    checks.push('balanced/budget/armor/min/max');
    await evaluate("document.getElementById('objectiveMode').scrollIntoView({block:'start'})");
    const balancedShot = await command('Page.captureScreenshot', {format:'png'});
    fs.writeFileSync(output + '/balanced-ui.png', Buffer.from(balancedShot.data, 'base64'));
    await evaluate("document.getElementById('objectiveMode').value='lexicographic'; document.getElementById('objectiveMode').dispatchEvent(new Event('change'))");
    const before = await evaluate('lexOrder.slice()');
    assert(before.length === 2, 'Active weighted goals');
    await evaluate("document.querySelector('#lexicographicOrder div button:last-child').click()");
    const order = await evaluate('lexOrder.slice()');
    assert(order[0] === before[1] && order[1] === before[0], 'Order controls');
    p = await clickBuild();
    assert(p.get('optimization_mode') === 'lexicographic' && JSON.stringify(JSON.parse(p.get('priority_order'))) === JSON.stringify(order), 'Explicit order');
    checks.push('lexicographic/reorder');
    await evaluate("document.getElementById('prio_health_bonus').value='0'; document.getElementById('prio_health_bonus').dispatchEvent(new Event('input',{bubbles:true}))");
    assert((await evaluate('lexOrder.slice()')).length === 1, 'Zero weight removed from order');
    await evaluate("document.getElementById('prio_health_bonus').value='8'; document.getElementById('prio_health_bonus').dispatchEvent(new Event('input',{bubbles:true})); document.getElementById('objectiveMode').value='min_budget'; document.getElementById('objectiveMode').dispatchEvent(new Event('change')); document.getElementById('min_sprint_speed_modifier').value='3'");
    p = await clickBuild();
    assert(p.get('optimization_mode') === 'min_budget' && !p.has('priority_order'), 'Minimum budget mode');
    checks.push('min_budget');
    await evaluate("document.getElementById('objectiveMode').value='balanced'; document.getElementById('objectiveMode').dispatchEvent(new Event('change')); document.getElementById('armorId').value='5lr71'; document.getElementById('armorId').dispatchEvent(new Event('change'))");
    p = await clickBuild(); assert(p.get('armor_id') === '5lr71', 'Armor change');
    await evaluate("document.getElementById('armorId').value=''; document.getElementById('armorId').dispatchEvent(new Event('change'))");
    p = await clickBuild(); assert(!p.has('armor_id'), 'Armor removal');
    checks.push('armor/change/remove');
    await evaluate("document.getElementById('armorId').value='3g6p5'; document.getElementById('metaBuildBtn').click()");
    await wait("!document.getElementById('metaBuildBtn').disabled");
    assert(last('/api/meta-builds').get('armor_id') === '3g6p5', 'Meta armor');
    await evaluate("document.getElementById('metaSweepBtn').click()");
    await wait("!document.getElementById('metaSweepBtn').disabled");
    assert(last('/api/budget-sweep').get('armor_id') === '3g6p5', 'Sweep armor');
    checks.push('meta/sweep/armor');
    await evaluate("document.getElementById('objectiveMode').value='lexicographic'; document.getElementById('objectiveMode').dispatchEvent(new Event('change')); document.querySelectorAll('[id^=prio_]').forEach(el => {el.value='0'; el.dispatchEvent(new Event('input',{bubbles:true}))})");
    const count = requests.length;
    await evaluate("document.getElementById('buildBtn').click()");
    await wait("!document.getElementById('buildBtn').disabled");
    assert(requests.length === count, 'Empty lexicographic goal should stay local');
    assert(await evaluate("document.getElementById('results').textContent.includes('положительный вес')"), 'Empty goal message');
    checks.push('empty/lexicographic');
    assert(errors.length === 0, 'Browser exceptions: ' + JSON.stringify(errors));
    const shot = await command('Page.captureScreenshot', {format:'png'});
    fs.writeFileSync(output + '/ui.png', Buffer.from(shot.data, 'base64'));
    fs.writeFileSync(output + '/ui.json', JSON.stringify({passed:true,checks,requests,errors}, null, 2));
    await command('Browser.close');
    ws.close();
})().catch(error => {console.error(error); ws.close(); process.exitCode=1;});
