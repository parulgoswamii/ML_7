
# PROJECT 1: CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING

import csv
import math
import random
import os


# 1. CREATE CUSTOMER DATASET

DATA_FILE = "customer_data.csv"
RESULT_FILE = "customer_segmentation_result.csv"


def create_dataset():
    random.seed(42)

    rows = []

    for i in range(1, 101):

        age = random.randint(18, 60)
        income = random.randint(20, 120)
        spending = random.randint(10, 100)
        frequency = random.randint(1, 30)

        rows.append([
            i,
            age,
            income,
            spending,
            frequency
        ])

    with open(DATA_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "CustomerID",
            "Age",
            "Annual_Income",
            "Spending_Score",
            "Purchase_Frequency"
        ])

        writer.writerows(rows)

    print("Customer dataset created successfully!")


# 2. LOAD DATASET

def load_dataset():

    data = []

    with open(DATA_FILE, "r") as file:

        reader = csv.DictReader(file)

        for row in reader:

            data.append({
                "CustomerID": int(row["CustomerID"]),
                "Age": float(row["Age"]),
                "Annual_Income": float(row["Annual_Income"]),
                "Spending_Score": float(row["Spending_Score"]),
                "Purchase_Frequency": float(row["Purchase_Frequency"])
            })

    return data


# 3. FEATURE SCALING

def standardize(data):

    features = [
        "Age",
        "Annual_Income",
        "Spending_Score",
        "Purchase_Frequency"
    ]

    columns = []

    for feature in features:

        values = [
            row[feature]
            for row in data
        ]

        mean = sum(values) / len(values)

        variance = sum(
            (x - mean) ** 2
            for x in values
        ) / len(values)

        std = math.sqrt(variance)

        if std == 0:
            std = 1

        scaled_column = [
            (x - mean) / std
            for x in values
        ]

        columns.append(scaled_column)

    scaled_data = []

    for i in range(len(data)):

        point = []

        for column in columns:
            point.append(column[i])

        scaled_data.append(point)

    return scaled_data


# 4. EUCLIDEAN DISTANCE

def distance(point1, point2):

    total = 0

    for a, b in zip(point1, point2):

        total += (a - b) ** 2

    return math.sqrt(total)


# 5. K-MEANS ALGORITHM

def kmeans(data, k, iterations=100):

    random.seed(42)

    # Select initial centroids
    centroids = random.sample(data, k)

    labels = [0] * len(data)

    for _ in range(iterations):

        # Assign points to nearest centroid
        for i, point in enumerate(data):

            distances = [
                distance(point, centroid)
                for centroid in centroids
            ]

            labels[i] = distances.index(
                min(distances)
            )

        # Calculate new centroids
        new_centroids = []

        for cluster in range(k):

            cluster_points = [
                data[i]
                for i in range(len(data))
                if labels[i] == cluster
            ]

            if len(cluster_points) == 0:

                new_centroids.append(
                    random.choice(data)
                )

                continue

            dimensions = len(data[0])

            new_centroid = []

            for d in range(dimensions):

                mean = sum(
                    point[d]
                    for point in cluster_points
                ) / len(cluster_points)

                new_centroid.append(mean)

            new_centroids.append(new_centroid)

        # Check convergence
        change = 0

        for old, new in zip(
            centroids,
            new_centroids
        ):

            change += distance(old, new)

        centroids = new_centroids

        if change < 0.001:

            break

    return labels, centroids


# 6. CALCULATE INERTIA

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


# 7. ELBOW METHOD

def elbow_method(data):

    print("\nElbow Method")
    print("------------")

    values = []

    for k in range(2, 9):

        labels, centroids = kmeans(
            data,
            k
        )

        inertia = calculate_inertia(
            data,
            labels,
            centroids
        )

        values.append(
            (k, inertia)
        )

        print(
            "K =", k,
            " Inertia =",
            round(inertia, 2)
        )

    # Save elbow values
    with open(
        "elbow_results.csv",
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Number_of_Clusters",
            "Inertia"
        ])

        writer.writerows(values)

    return values


# 8. SILHOUETTE SCORE

def silhouette_score(data, labels):

    clusters = sorted(
        set(labels)
    )

    scores = []

    for i in range(len(data)):

        current_cluster = labels[i]

        same_cluster = [
            j
            for j in range(len(data))
            if labels[j] == current_cluster
            and j != i
        ]

        if len(same_cluster) == 0:

            a = 0

        else:

            a = sum(
                distance(
                    data[i],
                    data[j]
                )
                for j in same_cluster
            ) / len(same_cluster)

        other_distances = []

        for cluster in clusters:

            if cluster == current_cluster:
                continue

            other_cluster = [
                j
                for j in range(len(data))
                if labels[j] == cluster
            ]

            if len(other_cluster) > 0:

                avg_distance = sum(
                    distance(
                        data[i],
                        data[j]
                    )
                    for j in other_cluster
                ) / len(other_cluster)

                other_distances.append(
                    avg_distance
                )

        if len(other_distances) == 0:

            b = 0

        else:

            b = min(other_distances)

        if max(a, b) == 0:

            score = 0

        else:

            score = (b - a) / max(a, b)

        scores.append(score)

    return sum(scores) / len(scores)


# 9. HIERARCHICAL CLUSTERING

def hierarchical_clustering(data, k):

    clusters = [
        [i]
        for i in range(len(data))
    ]

    while len(clusters) > k:

        minimum_distance = float("inf")

        merge_a = None
        merge_b = None

        for i in range(len(clusters)):

            for j in range(i + 1, len(clusters)):

                total_distance = 0
                count = 0

                for a in clusters[i]:

                    for b in clusters[j]:

                        total_distance += distance(
                            data[a],
                            data[b]
                        )

                        count += 1

                average_distance = (
                    total_distance / count
                )

                if average_distance < minimum_distance:

                    minimum_distance = average_distance

                    merge_a = i
                    merge_b = j

        clusters[merge_a].extend(
            clusters[merge_b]
        )

        del clusters[merge_b]

    labels = [0] * len(data)

    for cluster_number, cluster in enumerate(clusters):

        for index in cluster:

            labels[index] = cluster_number

    return labels


# 10. CREATE SVG VISUALIZATION

def create_visualization(
    data,
    labels,
    filename,
    title
):

    width = 900
    height = 600

    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2"
    ]

    # Income and spending are used for visualization
    x_values = [
        row[2]
        for row in data
    ]

    y_values = [
        row[3]
        for row in data
    ]

    min_x = min(x_values)
    max_x = max(x_values)

    min_y = min(y_values)
    max_y = max(y_values)

    def convert_x(x):

        return 70 + (
            (x - min_x) /
            (max_x - min_x)
        ) * 760

    def convert_y(y):

        return 520 - (
            (y - min_y) /
            (max_y - min_y)
        ) * 430

    with open(filename, "w") as file:

        file.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}">'
        )

        file.write(
            f'<rect width="{width}" '
            f'height="{height}" fill="white"/>'
        )

        file.write(
            f'<text x="450" y="35" '
            f'text-anchor="middle" '
            f'font-size="24" font-family="Arial">'
            f'{title}</text>'
        )

        # X axis
        file.write(
            '<line x1="70" y1="520" '
            'x2="830" y2="520" '
            'stroke="black"/>'
        )

        # Y axis
        file.write(
            '<line x1="70" y1="520" '
            'x2="70" y2="90" '
            'stroke="black"/>'
        )

        file.write(
            '<text x="450" y="570" '
            'text-anchor="middle" '
            'font-size="16" '
            'font-family="Arial">'
            'Annual Income</text>'
        )

        file.write(
            '<text x="20" y="300" '
            'transform="rotate(-90 20 300)" '
            'font-size="16" '
            'font-family="Arial">'
            'Spending Score</text>'
        )

        # Points
        for i in range(len(data)):

            x = convert_x(x_values[i])
            y = convert_y(y_values[i])

            color = colors[
                labels[i] % len(colors)
            ]

            file.write(
                f'<circle cx="{x}" cy="{y}" '
                f'r="6" fill="{color}"/>'
            )

        file.write("</svg>")

    print(
        "Visualization saved:",
        filename
    )


# 11. SAVE RESULTS


def save_results(
    original_data,
    kmeans_labels,
    hierarchical_labels
):

    with open(
        RESULT_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "CustomerID",
            "Age",
            "Annual_Income",
            "Spending_Score",
            "Purchase_Frequency",
            "KMeans_Cluster",
            "Hierarchical_Cluster"
        ])

        for i, row in enumerate(
            original_data
        ):

            writer.writerow([
                row["CustomerID"],
                row["Age"],
                row["Annual_Income"],
                row["Spending_Score"],
                row["Purchase_Frequency"],
                kmeans_labels[i],
                hierarchical_labels[i]
            ])

    print(
        "\nFinal result saved:",
        RESULT_FILE
    )


# 12. MAIN PROGRAM

def main():

    print("=" * 60)
    print("CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING")
    print("=" * 60)

    # Create dataset
    if not os.path.exists(DATA_FILE):

        create_dataset()

    else:

        print(
            "Existing customer dataset found."
        )

    # Load data
    customers = load_dataset()

    print(
        "\nTotal customers:",
        len(customers)
    )

    # Feature scaling
    scaled_data = standardize(
        customers
    )

    print(
        "Feature scaling completed."
    )

    # Elbow method
    elbow_method(
        scaled_data
    )

    # Optimal clusters
    k = 4

    print(
        "\nOptimal number of clusters selected:",
        k
    )

    # K-Means
    kmeans_labels, centroids = kmeans(
        scaled_data,
        k
    )

    print(
        "\nK-Means clustering completed."
    )

    # K-Means silhouette
    kmeans_silhouette = silhouette_score(
        scaled_data,
        kmeans_labels
    )

    print(
        "K-Means Silhouette Score:",
        round(kmeans_silhouette, 4)
    )

    # Cluster counts
    print(
        "\nK-Means Cluster Distribution:"
    )

    for cluster in range(k):

        count = kmeans_labels.count(
            cluster
        )

        print(
            "Cluster",
            cluster,
            ":",
            count,
            "customers"
        )

    # Hierarchical
    print(
        "\nRunning Hierarchical Clustering..."
    )

    hierarchical_labels = hierarchical_clustering(
        scaled_data,
        k
    )

    hierarchical_silhouette = silhouette_score(
        scaled_data,
        hierarchical_labels
    )

    print(
        "Hierarchical Silhouette Score:",
        round(
            hierarchical_silhouette,
            4
        )
    )

    # Comparison
    print(
        "\n" + "=" * 60
    )

    print(
        "CLUSTERING TECHNIQUE COMPARISON"
    )

    print(
        "=" * 60
    )

    print(
        "K-Means:",
        round(
            kmeans_silhouette,
            4
        )
    )

    print(
        "Hierarchical:",
        round(
            hierarchical_silhouette,
            4
        )
    )

    if kmeans_silhouette > hierarchical_silhouette:

        print(
            "\nK-Means performed better."
        )

    elif hierarchical_silhouette > kmeans_silhouette:

        print(
            "\nHierarchical Clustering performed better."
        )

    else:

        print(
            "\nBoth techniques performed similarly."
        )

    # Save result
    save_results(
        customers,
        kmeans_labels,
        hierarchical_labels
    )

    # Create visualizations
    create_visualization(
        scaled_data,
        kmeans_labels,
        "kmeans_clusters.svg",
        "Customer Segmentation - K-Means"
    )

    create_visualization(
        scaled_data,
        hierarchical_labels,
        "hierarchical_clusters.svg",
        "Customer Segmentation - Hierarchical"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "PROJECT 1 COMPLETED SUCCESSFULLY!"
    )

    print(
        "=" * 60
    )

    print(
        "\nOpen these files:"
    )

    print(
        "1. customer_data.csv"
    )

    print(
        "2. customer_segmentation_result.csv"
    )

    print(
        "3. elbow_results.csv"
    )

    print(
        "4. kmeans_clusters.svg"
    )

    print(
        "5. hierarchical_clusters.svg"
    )


# START PROGRAM

if __name__ == "__main__":
    main()