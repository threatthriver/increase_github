public class DataProcessor {
    private List<Integer> data = new ArrayList<>();

    public List<Integer> processData(List<Integer> items) {
        return items.stream().map(item -> item * 2).collect(Collectors.toList());
    }

    public double analyzeResults(List<Integer> data) {
        return data.stream().mapToInt(Integer::intValue).average().orElse(0.0);
    }
}