import csv
import math
import re
import os


DATA_FILE = "news_articles.csv"
RESULT_FILE = "news_clustering_result.csv"



def create_dataset():

    articles = [

        # TECHNOLOGY
        "Artificial intelligence is changing technology and software development",
        "New smartphone technology introduces advanced artificial intelligence features",
        "Technology companies are developing powerful artificial intelligence systems",
        "Cybersecurity experts discuss new security technologies",
        "Cloud computing and artificial intelligence are transforming businesses",
        "Software developers are using machine learning for modern applications",
        "New computer processors improve artificial intelligence performance",
        "Technology startups invest heavily in artificial intelligence research",

        # SPORTS
        "The cricket team won the international match after an exciting game",
        "Football players prepare for the upcoming championship tournament",
        "The cricket championship attracted thousands of sports fans",
        "Football team celebrates an important victory in the league",
        "Sports players train hard for the national tournament",
        "Cricket players scored a record number of runs in the match",
        "The football championship final was watched by millions of fans",
        "Athletes prepare for the international sports competition",

        # BUSINESS
        "Stock markets rise as investors respond to positive economic news",
        "Business companies report strong financial growth this year",
        "The central bank discusses interest rates and economic development",
        "New businesses attract investment from international companies",
        "Economic experts predict growth in the global market",
        "Companies increase their profits because of strong market demand",
        "Financial markets react to changes in interest rates",
        "Business leaders discuss investment and economic policies",

        # POLITICS
        "Government announces new policies for national development",
        "Political leaders discuss the upcoming national elections",
        "Parliament debates a new bill related to public services",
        "Government ministers meet to discuss national security",
        "Election campaigns begin across several states",
        "Political parties release their plans for the upcoming election",
        "The government introduces new public welfare policies",
        "Leaders discuss international relations and foreign policy"
    ]

    with open(
        DATA_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Article_ID",
            "Article"
        ])

        for i, article in enumerate(
            articles,
            start=1
        ):

            writer.writerow([
                i,
                article
            ])

    print(
        "News dataset created successfully!"
    )




def load_dataset():

    articles = []

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            articles.append({
                "Article_ID": int(
                    row["Article_ID"]
                ),
                "Article": row["Article"]
            })

    return articles


def clean_text(text):

    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    # Split into words
    words = text.split()

    # Common English stop words
    stop_words = {
        "the", "is", "a", "an", "and",
        "of", "to", "in", "for", "on",
        "with", "as", "this", "that",
        "are", "was", "were", "be",
        "by", "from", "at", "after",
        "new", "their", "its", "has",
        "have", "into", "about", "over"
    }

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return words


# ============================================================
# 4. TF-IDF
# ============================================================

def create_tfidf(documents):

    tokenized_documents = []

    for document in documents:

        words = clean_text(
            document
        )

        tokenized_documents.append(
            words
        )

    # Create vocabulary
    vocabulary = set()

    for words in tokenized_documents:

        for word in words:

            vocabulary.add(word)

    vocabulary = sorted(
        vocabulary
    )

    # Document frequency
    document_frequency = {}

    for word in vocabulary:

        count = 0

        for words in tokenized_documents:

            if word in words:
                count += 1

        document_frequency[word] = count

    total_documents = len(
        tokenized_documents
    )

    tfidf_matrix = []

    for words in tokenized_documents:

        vector = []

        for word in vocabulary:

            term_frequency = words.count(
                word
            )

            if len(words) > 0:

                tf = (
                    term_frequency /
                    len(words)
                )

            else:

                tf = 0

            df = document_frequency[word]

            idf = math.log(
                (total_documents + 1) /
                (df + 1)
            ) + 1

            tfidf = tf * idf

            vector.append(
                tfidf
            )

        tfidf_matrix.append(
            vector
        )

    return tfidf_matrix, vocabulary



def distance(point1, point2):

    total = 0

    for a, b in zip(
        point1,
        point2
    ):

        total += (
            (a - b) ** 2
        )

    return math.sqrt(
        total
    )



def kmeans(
    data,
    k,
    iterations=50
):

    # Initial centroids
    centroids = []

    step = len(data) // k

    for i in range(k):

        index = i * step

        centroids.append(
            data[index][:]
        )

    labels = [
        0
        for _ in data
    ]

    for iteration in range(
        iterations
    ):

        # Assign documents
        # to nearest centroid
        for i, point in enumerate(data):

            distances = []

            for centroid in centroids:

                distances.append(
                    distance(
                        point,
                        centroid
                    )
                )

            labels[i] = distances.index(
                min(distances)
            )

        # Calculate new centroids
        new_centroids = []

        dimensions = len(
            data[0]
        )

        for cluster in range(k):

            cluster_points = []

            for i in range(len(data)):

                if labels[i] == cluster:

                    cluster_points.append(
                        data[i]
                    )

            if len(cluster_points) == 0:

                new_centroids.append(
                    centroids[cluster]
                )

                continue

            new_centroid = []

            for d in range(dimensions):

                total = 0

                for point in cluster_points:

                    total += point[d]

                mean = (
                    total /
                    len(cluster_points)
                )

                new_centroid.append(
                    mean
                )

            new_centroids.append(
                new_centroid
            )

        # Check convergence
        change = 0

        for old, new in zip(
            centroids,
            new_centroids
        ):

            change += distance(
                old,
                new
            )

        centroids = new_centroids

        if change < 0.0001:

            break

    return labels, centroids


def calculate_inertia(
    data,
    labels,
    centroids
):

    total = 0

    for i, point in enumerate(data):

        cluster = labels[i]

        total += (
            distance(
                point,
                centroids[cluster]
            ) ** 2
        )

    return total


# ============================================================
# 8. ELBOW METHOD
# ============================================================

def elbow_method(data):

    print("\n")
    print("=" * 60)
    print("ELBOW METHOD")
    print("=" * 60)

    results = []

    for k in range(2, 7):

        labels, centroids = kmeans(
            data,
            k
        )

        inertia = calculate_inertia(
            data,
            labels,
            centroids
        )

        results.append([
            k,
            inertia
        ])

        print(
            "Clusters:",
            k,
            " | Inertia:",
            round(
                inertia,
                4
            )
        )

    with open(
        "news_elbow_results.csv",
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Number_of_Clusters",
            "Inertia"
        ])

        writer.writerows(
            results
        )

    return results



def silhouette_score(
    data,
    labels
):

    clusters = sorted(
        set(labels)
    )

    scores = []

    for i in range(
        len(data)
    ):

        current_cluster = labels[i]

        # Same cluster
        same_cluster = []

        for j in range(
            len(data)
        ):

            if (
                labels[j] ==
                current_cluster
                and i != j
            ):

                same_cluster.append(j)

        if len(same_cluster) == 0:

            a = 0

        else:

            total = 0

            for j in same_cluster:

                total += distance(
                    data[i],
                    data[j]
                )

            a = (
                total /
                len(same_cluster)
            )

        # Other clusters
        other_averages = []

        for cluster in clusters:

            if cluster == current_cluster:

                continue

            other_cluster = []

            for j in range(
                len(data)
            ):

                if labels[j] == cluster:

                    other_cluster.append(
                        j
                    )

            if len(other_cluster) > 0:

                total = 0

                for j in other_cluster:

                    total += distance(
                        data[i],
                        data[j]
                    )

                average = (
                    total /
                    len(other_cluster)
                )

                other_averages.append(
                    average
                )

        if len(other_averages) > 0:

            b = min(
                other_averages
            )

        else:

            b = 0

        denominator = max(
            a,
            b
        )

        if denominator == 0:

            score = 0

        else:

            score = (
                (b - a) /
                denominator
            )

        scores.append(
            score
        )

    return (
        sum(scores) /
        len(scores)
    )



def hierarchical_clustering(
    data,
    k
):

    # Initially every article
    # is its own cluster
    clusters = []

    for i in range(
        len(data)
    ):

        clusters.append([
            i
        ])

    while len(clusters) > k:

        smallest_distance = float(
            "inf"
        )

        merge_a = 0
        merge_b = 1

        # Find closest clusters
        for i in range(
            len(clusters)
        ):

            for j in range(
                i + 1,
                len(clusters)
            ):

                total = 0
                count = 0

                for a in clusters[i]:

                    for b in clusters[j]:

                        total += distance(
                            data[a],
                            data[b]
                        )

                        count += 1

                average = (
                    total /
                    count
                )

                if average < smallest_distance:

                    smallest_distance = (
                        average
                    )

                    merge_a = i
                    merge_b = j

        # Merge clusters
        clusters[merge_a].extend(
            clusters[merge_b]
        )

        del clusters[merge_b]

    labels = [
        0
        for _ in data
    ]

    for cluster_number, cluster in enumerate(
        clusters
    ):

        for index in cluster:

            labels[index] = (
                cluster_number
            )

    return labels



def display_clusters(
    articles,
    labels
):

    print("\n")
    print("=" * 60)
    print("NEWS ARTICLES GROUPED BY CLUSTER")
    print("=" * 60)

    clusters = sorted(
        set(labels)
    )

    for cluster in clusters:

        print(
            "\nCLUSTER",
            cluster
        )

        print("-" * 40)

        for i in range(
            len(articles)
        ):

            if labels[i] == cluster:

                print(
                    str(
                        articles[i][
                            "Article_ID"
                        ]
                    )
                    + ". "
                    + articles[i][
                        "Article"
                    ]
                )



def save_results(
    articles,
    kmeans_labels,
    hierarchical_labels
):

    with open(
        RESULT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Article_ID",
            "Article",
            "KMeans_Cluster",
            "Hierarchical_Cluster"
        ])

        for i in range(
            len(articles)
        ):

            writer.writerow([
                articles[i][
                    "Article_ID"
                ],

                articles[i][
                    "Article"
                ],

                kmeans_labels[i],

                hierarchical_labels[i]
            ])

    print(
        "\nResult saved as:",
        RESULT_FILE
    )


def create_visualization(
    articles,
    labels,
    filename,
    title
):

    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2"
    ]

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("""
<!DOCTYPE html>
<html>
<head>
<title>News Clustering</title>

<style>

body {
    font-family: Arial;
    background: #f5f7fb;
    padding: 30px;
}

h1 {
    text-align: center;
}

.container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.card {
    width: 280px;
    padding: 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.15);
}

.cluster {
    font-size: 18px;
    font-weight: bold;
}

.article {
    margin-top: 10px;
    line-height: 1.5;
}

</style>

</head>

<body>
""")

        file.write(
            "<h1>" +
            title +
            "</h1>"
        )

        file.write(
            '<div class="container">'
        )

        for i in range(
            len(articles)
        ):

            color = colors[
                labels[i] %
                len(colors)
            ]

            file.write(
                '<div class="card" '
                'style="border-top: 5px solid '
                + color +
                ';">'
            )

            file.write(
                '<div class="cluster">'
                "Cluster "
                + str(labels[i])
                + "</div>"
            )

            file.write(
                '<div class="article">'
                + str(
                    articles[i][
                        "Article"
                    ]
                )
                + "</div>"
            )

            file.write(
                "</div>"
            )

        file.write(
            "</div></body></html>"
        )

    print(
        "Visualization saved:",
        filename
    )




def main():

    print("=" * 60)

    print(
        "NEWS ARTICLE CLUSTERING SYSTEM"
    )

    print("=" * 60)

    # Create dataset
    if not os.path.exists(
        DATA_FILE
    ):

        create_dataset()

    else:

        print(
            "Existing news dataset found."
        )

    # Load data
    articles = load_dataset()

    print(
        "\nTotal articles:",
        len(articles)
    )

    # Text preprocessing
    documents = []

    for article in articles:

        documents.append(
            article["Article"]
        )

    print(
        "Text preprocessing completed."
    )

    # TF-IDF
    tfidf_matrix, vocabulary = create_tfidf(
        documents
    )

    print(
        "TF-IDF feature extraction completed."
    )

    print(
        "Total vocabulary features:",
        len(vocabulary)
    )

    # Elbow
    elbow_method(
        tfidf_matrix
    )

    # Number of clusters
    k = 4

    print(
        "\nSelected number of clusters:",
        k
    )

    # K-Means
    kmeans_labels, centroids = kmeans(
        tfidf_matrix,
        k
    )

    print(
        "\nK-Means clustering completed."
    )

    # K-Means score
    kmeans_score = silhouette_score(
        tfidf_matrix,
        kmeans_labels
    )

    print(
        "K-Means Silhouette Score:",
        round(
            kmeans_score,
            4
        )
    )

    # Display clusters
    display_clusters(
        articles,
        kmeans_labels
    )

    # Hierarchical clustering
    print(
        "\nRunning Hierarchical Clustering..."
    )

    hierarchical_labels = hierarchical_clustering(
        tfidf_matrix,
        k
    )

    print(
        "Hierarchical clustering completed."
    )

    # Hierarchical score
    hierarchical_score = silhouette_score(
        tfidf_matrix,
        hierarchical_labels
    )

    print(
        "Hierarchical Silhouette Score:",
        round(
            hierarchical_score,
            4
        )
    )

    # Comparison
    print("\n")
    print("=" * 60)
    print("CLUSTERING TECHNIQUE COMPARISON")
    print("=" * 60)

    print(
        "K-Means Silhouette Score:",
        round(
            kmeans_score,
            4
        )
    )

    print(
        "Hierarchical Silhouette Score:",
        round(
            hierarchical_score,
            4
        )
    )

    if kmeans_score > hierarchical_score:

        print(
            "\nK-Means performed better."
        )

    elif hierarchical_score > kmeans_score:

        print(
            "\nHierarchical Clustering performed better."
        )

    else:

        print(
            "\nBoth methods performed similarly."
        )

    # Save results
    save_results(
        articles,
        kmeans_labels,
        hierarchical_labels
    )

    # Create visualizations
    create_visualization(
        articles,
        kmeans_labels,
        "news_kmeans_visualization.html",
        "News Article Clustering - K-Means"
    )

    create_visualization(
        articles,
        hierarchical_labels,
        "news_hierarchical_visualization.html",
        "News Article Clustering - Hierarchical"
    )

    print("\n")
    print("=" * 60)
    print(
        "PROJECT 2 COMPLETED SUCCESSFULLY!"
    )
    print("=" * 60)

    print("\nGenerated files:")

    print("1. news_articles.csv")
    print("2. news_clustering_result.csv")
    print("3. news_elbow_results.csv")
    print("4. news_kmeans_visualization.html")
    print("5. news_hierarchical_visualization.html")



# START PROGRAM


if __name__ == "__main__":

    main()