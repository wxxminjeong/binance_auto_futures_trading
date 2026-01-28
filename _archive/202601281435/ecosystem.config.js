module.exports = {
  apps: [
    {
      name: "COINBOT",
      script: "coinbot_v6.py",
      interpreter: "python3",
      autorestart: true
    },
    {
      name: "WEB",
      script: "log_server.py",
      interpreter: "python3",
      autorestart: true
    }
  ]
};