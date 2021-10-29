try {
  require('puppeteer')
} catch (e) {
  console.error(
"\nFATAL ERROR:" +
"\n    Package 'puppeteer' is not available." +
"\n    Run 'npm install --no-save puppeteer' before running this script" +
"\n    * You may use PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=1 environment variable to avoid automatic Chromium downloading" +
"\n      (specify own Chromium/Chrome version through PUPPETEER_EXECUTABLE_PATH=`which google-chrome` environment variable)" +
"\n");
  process.exit(1);
}
const puppeteer = require('puppeteer')
const colors = require("ansi-colors")
const path = require("path");
const fs = require("fs");
const http = require("http");

run_main(require('minimist')(process.argv.slice(2)));

async function run_main(o = {}) {
  try {
    await main(o);
    console.magenta("FATAL: Unexpected exit!");
    process.exit(1);
  } catch (e) {
    console.error(colors.magenta("FATAL: Unexpected exception!"));
    console.error(e);
    process.exit(1);
  }
}

async function main(o = {}) {
  o = Object.assign({}, {
    buildFolder: __dirname,
    port: 8080,
    debug: false,
    noHeadless: false,
    serverPrefix: `http://localhost`,
    noExit: false,
    screenshot: undefined,
    help: false,
    noTryCatch: false,
    maxBlockDuration: 30000
  }, o)
  if (typeof o.screenshot == 'string' && o.screenshot == 'false') {
    console.log(colors.red('ERROR: misused screenshot option, use --no-screenshot instead'));
  }
  if (o.noExit) {
    o.maxBlockDuration = 999999999
  }
  o.debug && console.log('Current Options', o);
  if (o.help) {
    printHelpAndExit();
  }
  const serverAddress = `${o.serverPrefix}:${o.port}`
  const url = `${serverAddress}/tests.html${o.noTryCatch ? '?notrycatch=1' : ''}`;
  if (!fs.existsSync(o.buildFolder)) {
    console.error(`Expected folder "${o.buildFolder}" to exists. Aborting`);
  }
  o.debug && debug('Server Listening at ' + url);
  const server = await staticServer(o.buildFolder, o.port, m => debug, m => error);
  o.debug && debug(`Browser launching ${!o.noHeadless ? 'headless' : 'not headless'}`);
  const browser = await puppeteer.launch({ headless: !o.noHeadless });
  const page = await browser.newPage();
  page.on('console', e => {
    locationMsg = formatMessage(`${e.location().url}:${e.location().lineNumber}:${e.location().columnNumber}`);
    if (e.type() === 'error') {
      console.log(colors.red(formatMessage('' + e.text(), `-- ERROR:${locationMsg}: `, )));
    }
    else if (o.debug) {
      o.debug && console.log(colors.grey(formatMessage('' + e.text(), `-- ${locationMsg}: `)));
    }
  });
  o.debug && debug(`Opening page address ${url}`);
  await page.goto(url);
  await page.waitForFunction(() => (document.querySelector(`#qunit-testresult`) && document.querySelector(`#qunit-testresult`).textContent || '').trim().toLowerCase().startsWith('tests completed'));
  const text = await getText(`#qunit-testresult`);
  if (!text) {
    return await fail(`An error occurred extracting test results. Check the build folder ${o.buildFolder} is correct and has build with tests enabled.`);
  }
  o.debug && debug(colors.blackBright("* UserAgent: " + await getText('#qunit-userAgent')));
  const testFailed = !text.includes(' 0 failed');
  if (testFailed && !o.debug) {
    process.stdout.write(colors.grey("* Use '--debug' parameter to see details of failed tests.\n"));
  }
  if (o.screenshot || (o.screenshot === undefined && testFailed)) {
    await page.screenshot({ path: 'screenshot.png', fullPage: 'true' });
    process.stdout.write(colors.grey(`* Screenshot taken: ${o.buildFolder}/screenshot.png\n`));
  }
  if (testFailed) {
    const report = await failReport();
    process.stdout.write(`
${colors.red.bold.underline('Failed tests ! :(')}

${colors.redBright(colors.symbols.cross + ' ' + report.join(`\n${colors.symbols.cross} `))}

${colors.redBright(`=== Summary ===\n${text}`)}
`);
  }
  else {
    process.stdout.write(colors.green(`
 ${colors.symbols.check} No Errors :)

 === Summary ===\n${text}
`));
  }
  if (o.noExit) {
    while (true) {
      await new Promise(r => setTimeout(r, 5000));
    }
  }
  await server && server.close();
  await browser.close();
  process.exit(testFailed ? 1 : 0);

  async function getText(s) {
    return await page.evaluate((s) => (document.querySelector(s) && document.querySelector(s).innerText) || ''.trim(), s);
  }
  async function failReport() {
    const failures = await page.evaluate(() => Array.from(document.querySelectorAll('#qunit-tests .fail')).filter(e => e.querySelector('.module-name')).map(e => ({
      moduleName: e.querySelector('.module-name') && e.querySelector('.module-name').textContent,
      testName: e.querySelector('.test-name') && e.querySelector('.test-name').textContent,
      expected: e.querySelector('.test-expected pre') && e.querySelector('.test-expected pre').textContent,
      actual: e.querySelector('.test-actual pre') && e.querySelector('.test-actual pre').textContent,
      code: e.querySelector('.test-source') && e.querySelector('.test-source').textContent.replace("Source:     at ", ""),
    })));
    return failures.map(f => `${f.moduleName}: ${f.testName} (${formatMessage(f.code)})`);
  }
  async function fail(s) {
    await failReport();
    process.stdout.write(colors.red(s) + '\n');
    if (o.screenshot || o.screenshot === undefined) {
      await page.screenshot({ path: 'screenshot.png', fullPage: 'true' });
      process.stdout.write(colors.grey(`* Screenshot taken: ${o.buildFolder}/screenshot.png\n`));
    }
    process.exit(1);
  }
  async function debug(s) {
    process.stdout.write(s + '\n');
  }
  async function error(s) {
    process.stdout.write(s + '\n');
  }
  function formatMessage(message, prefix) {
    prefix = prefix || '';
    return prefix + ('' + message).split('\n').map(l => l.replace(serverAddress, o.buildFolder)).join('\n' + prefix);
  }
}


function printHelpAndExit() {
  console.log(`
Usage:

  # First, remember to build opencv.js with tests enabled:
  ${colors.blueBright(`python ./platforms/js/build_js.py build_js --build_test`)}

  # Install the tool locally (needed only once) and run it
  ${colors.blueBright(`cd build_js/bin`)}
  ${colors.blueBright(`npm install`)}
  ${colors.blueBright(`node run_puppeteer`)}

By default will run a headless browser silently printing a small report in the terminal.
But it could used to debug the tests in the browser, take screenshots, global tool or
targeting external servers exposing the tests.

TIP: you could install the tool globally (npm install --global build_js/bin) to execute it from any local folder.

# Options

 * port?: number. Default 8080
 * buildFolder?: string. Default __dirname (this folder)
 * debug?: boolean. Default false
 * noHeadless?: boolean. Default false
 * serverPrefix?: string . Default http://localhost
 * help?: boolean
 * screenshot?: boolean . Make screenshot on failure by default. Use --no-screenshot to disable screenshots completely.
 * noExit?: boolean default false. If true it will keep running the server - together with noHeadless you can debug in the browser.
 * noTryCatch?: boolean will disable Qunit tryCatch - so exceptions are dump to stdout rather than in the browser.
 * maxBlockDuration: QUnit timeout. If noExit is given then is infinity.
  `);
  process.exit(0);
}

async function staticServer(basePath, port, onFound, onNotFound) {
  return new Promise(async (resolve) => {
    const server = http.createServer((req, res) => {
      var url = resolveUrl(req.url);
      onFound && onFound(url);
      var stream = fs.createReadStream(path.join(basePath, url || ''));
      stream.on('error', function () {
        onNotFound && onNotFound(url);
        res.writeHead(404);
        res.end();
      });
      stream.pipe(res);
    }).listen(port);
    server.on('listening', () => {
      resolve(server);
    });
  });
  function resolveUrl(url = '') {
    var i = url.indexOf('?');
    if (i != -1) {
      url = url.substr(0, i);
    }
    i = url.indexOf('#');
    if (i != -1) {
      url = url.substr(0, i);
    }
    return url;
  }
}
