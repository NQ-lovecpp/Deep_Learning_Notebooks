const axios = require('axios');

axios.post('http://localhost:11434/v1/completions', {
  model: 'deepseek-r1:7b',  // 根据实际情况调整
  prompt: '你好，请生成一段文字。',
  // 可以根据需要增加其他参数
}, {
  headers: {
    'Content-Type': 'application/json'
    // 如果有验证需求，添加相关的头信息
  }
})
.then(response => {
  console.log('模型返回结果：', response.data);
})
.catch(error => {
  console.error('请求失败：', error);
});
