// Modules to control application life and create native browser window
const {app, BrowserWindow} = require('electron')
const path = require('path')

// Keep a global reference of the window object, if you don't, the window will
// be closed automatically when the JavaScript object is garbage collected.
let mainWindow = {}

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1220,
    height: 840,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: app.getAppPath()+"/node_setup.js"
    }
  })

  // Load the index.html with 'numRunsParm' to run inference multiple times.
  let url = `file://${__dirname}/js_image_classification_webnn_electron.html`
  const numRunsParm = '?' + process.argv[2]
  mainWindow.loadURL(url + numRunsParm)

  // Emitted when the window is closed.
  mainWindow.on('closed', function() {
    // Dereference the window object, usually you would store windows
    // in an array if your app supports multi windows, this is the time
    // when you should delete the corresponding element.
    mainWindow = null
  })
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.on('ready', createWindow)

// Quit when all windows are closed.
app.on('window-all-closed', function() {
  // On macOS it is common for applications and their menu bar
  // to stay active until the user quits explicitly with Cmd + Q
  if (process.platform !== 'darwin') app.quit()
})

app.on(
    'activate',
    function() {
      // On macOS it's common to re-create a window in the app when the
      // dock icon is clicked and there are no other windows open.
      if (mainWindow === null) createWindow()
    })

    // In this file you can include the rest of your app's specific main process
    // code. You can also put them in separate files and require them here.
