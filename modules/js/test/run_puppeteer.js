const puppeteer= require('puppeteer')
const colors = require("ansi-colors")
const path = require("path");
const fs = require("fs");
const http = require("http");
const {inspect} = require('util')

main(require('minimist')(process.argv.slice(2)));

async function main(o={}) {
o = Object.assign({} ,{  
  buildFolder: __dirname,
  port: 8080,
  debug: true,
  noHeadless: false,
  serverPrefix: `http://localhost`,
  noExit: false,
  help: false,
  noTryCatch: false,
  maxBlockDuration: 30000}, o)
  if(o.noExit){
o.maxBlockDuration = 999999999
  }

o.debug && console.log('Current Options', o);

  if (o.help) {
    printHelpAndExit();
  }
  const serverAddress = `${o.serverPrefix}:${o.port}`
  const url = `${serverAddress}/tests.html${o.noTryCatch ? '?notrycatch=1' : ''}`;
  // const folder = path.resolve(`${o.buildFolder}`);
  if (!fs.existsSync(o.buildFolder)) {
    console.error(`Expected folder "${o.buildFolder} to exists. Aborting`);
  }
  o.debug && debug('Server Listening at ' + url);
  const server = await staticServer(o.buildFolder, o.port, m => debug, m => error);
  o.debug && debug(`Browser launching ${!o.noHeadless ? 'headless' : 'not headless'}`);
  const browser = await puppeteer.launch({ headless: !o.noHeadless });
  const page = await browser.newPage();
  // page.on('º')
  // page.browserContext().browser().on('')
  // await page.evaluate(() => {
  //   window.addEventListener('error', (message, file, lineNumber, columnNumber, code, error) => {
  //     // debugger
  //     console.error(message, file, lineNumber, columnNumber, code, error && error.stack, Object.keys(error));
      
  //     // console.log.apply(message)
  //   })
  // });
  // page.on('error', e => {
  //   // debugger
  //   // e && error('DOM Error: '+e+', '+e.name+', '+e.message+', '+e.stack);
  //   // e && error('DOM Error: '+e+', '+e.name+', '+e.message+', '+e.stack);
  // });
  // page.on('pageerror', function (e) {
  //     // console.log('error, arguments', e)

  //   // e && error('Page Error: '+e+', '+e.name+', '+e.message+', '+e.stack);
  // });
  page.on('console', e => {
    if (e.type() === 'error') {
      // debugger
      console.log('error: ',e.location(), e.text().split('\n').map(l=>l.replace(serverAddress, o.buildFolder)).join('\n'))
        // console.log([e.text(), e.location() ? JSON.stringify(e.location()):'', (e.args()||[]).map(e => inspect(e))].join(', '));
    }
    else if (o.debug) {
       o.debug && console.log('log: ',e.location(), e.text())

  // console.log([e.text(), e.location() ? JSON.stringify(e.location()):'', (e.args()||[]).map(e => inspect(e))].join(', '));
    }
  });
  o.debug && debug(`Opening page address ${url}`);
  await page.goto(url);
  await page.waitForFunction(() => (document.querySelector(`#qunit-testresult`) && document.querySelector(`#qunit-testresult`).textContent || '').trim().toLowerCase().startsWith('tests completed'));
  const text = await getText(`#qunit-testresult`);
  if (!text) {
    return await fail(`An error occurred extracting test results. Check the build folder ${o.buildFolder} is correct and has build with tests enabled.`);
  }
  o.debug && debug(await getText('#qunit-userAgent'));
  const testFailed = !text.includes('0 failed');
  if (testFailed) {
    const report = await failReport();
    process.stdout.write(`
${colors.red.bold.underline('Failed tests ! :(')}

${colors.redBright(colors.symbols.cross + ' ' + report.join(`\n ${colors.symbols.cross} `))}

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
      textName: e.querySelector('.test-name') && e.querySelector('.test-name').textContent,
      expected: e.querySelector('.test-expected pre') && e.querySelector('.test-expected pre').textContent,
      actual: e.querySelector('.test-actual pre') && e.querySelector('.test-actual pre').textContent,
      code: e.querySelector('.qunit-source') && e.querySelector('.qunit-source').textContent,
    })));
    return failures.map(f => `${f.moduleName}:`);
  }
  async function fail(s) {
    await failReport();
    await page.screenshot({ path: 'example.png' });
    process.stdout.write(colors.grey(s + `
 * Screenshot taken: ${o.buildFolder}/tmp_screenshot.png\n`));
    process.exit(1);
  }
  async function debug(s) {
    process.stdout.write(colors.blackBright(s + '\n'));
  }
  async function error(s) {
    process.stdout.write(colors.redBright(s + '\n'));
  }
}


function printHelpAndExit() {
  console.log(`
Usage: 

  # Install it globally (only needed once):
  ${colors.blueBright(`npm install --global opencv/platforms/js/test-runner`)}

  # Build opencv with the tests tests:
  ${colors.blueBright(`python ./platforms/js/build_js.py build_js --build_test`)}

  # Run the tests
  ${colors.blueBright(`opencvjs-run-test --buildFolder build_js --port 8081 --debug`)}
  
# Options
 * port?: number
 * buildFolder?: string
 * debug?: boolean
 * headless?: boolean
 * keepServer?: boolean
 * help?: boolean

Note: for running the tool locally, you just install it without passing --global and running from 
its folder:

  ${colors.blueBright(`cd $HOME/opencv/platforms/js/test-runner`)}
  ${colors.blueBright(`npm install`)}
  ${colors.blueBright(`node bin/opencvjs-run-test $HOME/opencv/build_js`)}

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
