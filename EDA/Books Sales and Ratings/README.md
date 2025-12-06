
# 📚 Book Sales Data Analysis (EDA)

This project performs a full **Exploratory Data Analysis (EDA)** on a dataset of books that includes sales performance, ratings, authors, genres, publishing years, and revenue-related attributes.  
The goal is to uncover meaningful insights about **book performance**, **author growth**, **genre popularity**, and **sales trends**.

---

## 📁 Dataset Overview

The dataset **Books_Data_Clean.csv** contains detailed information about books including:

| Column Name | Description |
|-------------|-------------|
| Publishing Year | Year the book was published |
| Book Name | Title of the book |
| Author | Author of the book |
| language_code | Language code of the book |
| Author_Rating | Rating of the author |
| Book_average_rating | Average rating of the book |
| Book_ratings_count | Number of ratings |
| genre | Book genre |
| gross sales | Total sales revenue |
| publisher revenue | Publisher earnings |
| sale price | Price at which the book was sold |
| sales rank | Rank based on sale performance |
| Publisher | Name of the publisher |
| units sold | Total units sold |

Dataset Shape:

```

(1070 rows, 14 columns)

````

---

## 📦 Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
````

---

## 🧹 Data Cleaning

### ✔ Dropped unwanted column

`index` column removed.

### ✔ Removed entries with invalid publishing years

Filtered Publishing Year > 1900.

### ✔ Missing Values

* Book Name → 21 missing values (removed)
* language_code → 47 missing values (kept)

### ✔ Duplicate rows

```
Duplicated rows → 0
```

### ✔ Dataset unique values

Dataset includes:

* 987 unique book names
* 669 unique authors
* 8 languages
* 4 genres
* 9 publishers

---

## 📊 Exploratory Data Analysis (EDA)

### 🔹 1. Publishing Year Distribution

Shows publishing activity over time.

**Plot: Histogram**

```python
plt.hist(Book_Data["Publishing Year"])
```

Interpretation:

* Most books published between **1980 and 2015**
* Very few before 1960

---

### 🔹 2. Genre Distribution

**Plot: Countplot**

```python
sns.countplot(data=Book_Data, x="genre")
```

Genre counts:

* genre fiction: **759**
* nonfiction: **160**
* fiction: **54**
* children: **15**

📌 **Fiction dominates the dataset heavily.**

---

### 🔹 3. Top-Rated Books (by average rating)

```python
Book_Data.groupby("Book Name")["Book_average_rating"].mean().sort_values(ascending=False).head()
```

Top books include:

* *Words of Radiance*
* *A Court of Mist and Fury*
* *Calvin and Hobbes*
* *The Way of Kings*

---

### 🔹 4. Ratings Count Across Genres

**Plot: Boxplot**

```python
sns.boxplot(data=Book_Data, x="genre", y="Book_ratings_count")
```

Insights:

* Fiction and nonfiction books receive **more ratings**
* Children’s books have **lower engagement**

---

### 🔹 5. Relationship Between Sale Price & Units Sold

**Plot: Scatterplot**

```python
sns.scatterplot(data=Book_Data, x="sale price", y="units sold")
```

Insight:

* When **sale price is low**, **units sold increases**
* Indicates price sensitivity among readers

---

### 🔹 6. Language Distribution

**Plot: Countplot**

```python
sns.countplot(data=Book_Data, x="language_code")
```

Most used languages:

* eng: 670
* en-US: 226
* en-GB: 29

📌 English dominates book sales.

---

### 🔹 7. Publishers With Highest Revenue

```python
Book_Data.groupby("Publisher")["publisher revenue"].sum().sort_values(ascending=False)
```

Top publishers:

| Publisher                | Total Revenue |
| ------------------------ | ------------- |
| Penguin Group (USA) LLC  | 191,581       |
| Random House LLC         | 174,956       |
| Amazon Digital Services  | 141,767       |
| HarperCollins Publishers | 121,769       |

📌 Penguin Group leads the market.

---

### 🔹 8. Author Rating Distribution

```python
Book_Data.Author_Rating.value_counts()
```

| Rating       | Count |
| ------------ | ----- |
| Intermediate | 576   |
| Excellent    | 336   |
| Famous       | 48    |
| Novice       | 28    |

---

### 🔹 9. Authors With Highest Gross Sales

```python
growth_Sales = Book_Data.groupby("Author")["gross sales"].max()
```

Top authors:

| Author                 | Gross Sales |
| ---------------------- | ----------- |
| Harper Lee             | 47,795      |
| David Sedaris          | 41,250      |
| Laini Taylor           | 37,952      |
| Unknown, Seamus Heaney | 34,160      |
| Charles Duhigg         | 27,491      |

**Plot: Bar Chart**

```python
growth_Sales.head().plot(kind="bar")
```

---

### 🔹 10. Book Names With Highest Sales Rank

```python
Book_Data.groupby("Book Name")["sales rank"].max().sort_values(ascending=False).head()
```

---

### 🔹 11. Units Sold Over Years

**Plot: Line Chart**

```python
Book_Data.groupby("Publishing Year")["units sold"].sum().plot(kind="line")
```

### Interpretation:

The industry shows:

📈 **Slow growth (1900–1950)**
📈 **Strong growth (1950–1980)**
🚀 **Rapid expansion (1980–2015)**

After 2015, slight decline due to:

* Digital books
* Market saturation
* Changing reading habits

---

## 🧠 Key Insights Summary

✔ Fiction is the most popular genre
✔ English is the dominant language
✔ Lower prices → higher units sold
✔ Penguin Group is the highest-earning publisher
✔ Harper Lee shows the highest single-book sales
✔ Ratings and sales positively influence performance
✔ Book publishing peaked between **1980–2015**

---

## 🏁 Conclusion

This EDA provides a complete breakdown of the book sales market, identifying key factors that drive:

* Higher revenue
* Greater units sold
* Higher author success
* Reader preferences

The analysis can support:

* Publisher decision-making
* Marketing strategies
* Price optimization
* Author performance evaluation

---



