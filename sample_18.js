class DataProcessor {
    constructor() {
        this.data = [];
    }

    processData(items) {
        return items.map(item => item * 2);
    }

    analyzeResults(data) {
        return data.length ? data.reduce((a, b) => a + b) / data.length : 0;
    }
}

const processor = new DataProcessor();
const testData = Array.from({length: 5}, () => Math.floor(Math.random() * 100));
const results = processor.processData(testData);
const average = processor.analyzeResults(results);
console.log(`Processed data: ${results}`);
console.log(`Average: ${average}`);