// utils/logger.js - 日誌管理

export class Logger {
  static info(message, ...args) {
    console.log(`[INFO] ${message}`, ...args);
  }

  static success(message, ...args) {
    console.log(`[OK] ${message}`, ...args);
  }

  static warning(message, ...args) {
    console.warn(`[WARN] ${message}`, ...args);
  }

  static error(message, ...args) {
    console.error(`[ERROR] ${message}`, ...args);
  }

  static debug(message, ...args) {
    if (process.env.LOG_LEVEL === 'debug') {
      console.log(`[DEBUG] ${message}`, ...args);
    }
  }

  static tool(message, ...args) {
    console.log(`[TOOL] ${message}`, ...args);
  }

  static api(message, ...args) {
    console.log(`[API] ${message}`, ...args);
  }

  static response(message, ...args) {
    console.log(`[RESP] ${message}`, ...args);
  }

  static server(message, ...args) {
    console.log(`[SERVER] ${message}`, ...args);
  }
}
