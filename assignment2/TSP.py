import math
import random


class Pair:
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def getKey(self):
        return self.key

    def getValue(self):
        return self.value


class Node:
    def __init__(self, name, location):
        self.name = name
        self.visited = False
        self.location = location
        self.neighbors = []

    def setVisited(self, isVisited):
        self.visited = isVisited

    def isVisited(self):
        return self.visited

    def addNeighbor(self, neighbor):
        self.neighbors.append(neighbor)

    def getNeighbors(self):
        return self.neighbors

    def getName(self):
        return self.name

    def getLocation(self):
        return self.location


class Graph:
    def __init__(self):
        self.nodes = []
        self.nodeDict = {}

    def addNode(self, node):
        self.nodes.append(node)
        self.nodeDict[node.getName()] = node

    def createEdge(self, start, end):
        start.addNeighbor(end)
        end.addNeighbor(start)

    def getNodes(self):
        return self.nodes

    def getNodeByName(self, name):
        return self.nodeDict[name]

    def allNodesVisited(self):
        for node in self.nodes:
            if not node.isVisited():
                return False
        return True


class TravelingSalesmanProblem:
    def __init__(self, graph):
        self.distances = {}
        self.graph = graph

    def solve(self, strategy):

        self.calculateDistances(self.graph)

        if strategy is 0:
            self.hillClimbMethod()
        elif strategy is 1:
            self.geneticMethod()
        else:
            print("Unknown strategy")

    def calculateDistances(self, graph):
        for startingNode in graph.getNodes():
            self.distances[startingNode.getName()] = []

            for desinationNode in graph.getNodes():
                if startingNode.getName() is desinationNode.getName():
                    continue

                distance = self.distBetweenNodes(startingNode, desinationNode)
                cityLocationPair = Pair(desinationNode.getName(), distance)
                self.distances[startingNode.getName()].append(cityLocationPair)

    def distBetweenNodes(self, nodeA, nodeB):
        x1 = nodeA.getLocation()[0]
        x2 = nodeB.getLocation()[0]
        y1 = nodeA.getLocation()[1]
        y2 = nodeB.getLocation()[1]

        return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

    def getClosestUnvistedNode(self, node):
        distanceToNodes = self.distances[node.getName()]

        closest = Pair("temp", float("inf"))

        for pair in distanceToNodes:
            if pair.getValue() < closest.getValue() and not self.graph.getNodeByName(pair.getKey()).isVisited():
                closest = pair

        return closest.getKey(), closest.getValue()

    def hillClimbMethod(self):
        cities = self.graph.getNodes()
        dist = 0
        path = ""

        startingCity = cities[0]
        currCity = startingCity

        while True:
            nextCityName, distToNextCity = self.getClosestUnvistedNode(currCity)

            path = path + currCity.getName() + ", "

            if self.graph.allNodesVisited() or nextCityName is "temp":
                break

            dist = dist + distToNextCity

            currCity.setVisited(True)
            currCity = self.graph.getNodeByName(nextCityName)

        path = path + startingCity.getName()
        dist = dist + self.distBetweenNodes(startingCity, currCity)

        print("Hill Climb Finished")
        print("Total Distance: ", dist)
        print("Path: ", path)

    def routeDistance(self, orderedCityRoute):
        dist = 0
        path = ""

        for i in range(0, len(orderedCityRoute) - 1):
            currCity = orderedCityRoute[i]
            i += 1
            nextCity = orderedCityRoute[i]

            path = path + currCity.getName() + ", "
            dist = dist + self.distBetweenNodes(currCity, nextCity)

        path += orderedCityRoute[len(orderedCityRoute)-1].getName()

        return dist

    def routeFitness(self, orderedCityRoute):
        return 1 / (float(self.routeDistance(orderedCityRoute)))

    def createRandomListsOfOrderedCities(self, populationSize):
        cities = self.graph.getNodes()
        population = []

        for i in range(0, populationSize):
            population.append(random.sample(cities, len(cities)))

        return population

    def rankRoutesByFitness(self, routes):
        return sorted(routes, key=self.routeFitness)

    def selection(self, rankedPopulation):
        selectedRoutes = []
        selectedRoutes.append(rankedPopulation[0])
        selectedRoutes.append(rankedPopulation[1])
        randomSelectionIndex = random.randint(2, len(rankedPopulation) - 1)
        selectedRoutes.append(rankedPopulation[randomSelectionIndex])
        loopBreak = 10000
        while True:
            temp = random.randint(2, len(rankedPopulation) - 1)

            if temp is not randomSelectionIndex:
                selectedRoutes.append(rankedPopulation[temp])
                break

            loopBreak = loopBreak - 1
            if loopBreak is 0:
                break

        return selectedRoutes

    def crossover(self, selectionParents):
        randomizedSample = random.sample(selectionParents, len(selectionParents))
        children = []

        for i in range(0, len(randomizedSample)):
            aChild = []

            parentOne = randomizedSample[i]
            parentTwo = randomizedSample[len(randomizedSample) - i - 1]

            tempA = int(random.random() * len(parentOne))
            tempB = int(random.random() * len(parentOne))

            startingCityIndex = min(tempA, tempB)
            endingCityIndex = max(tempA, tempB)

            cityNames = []

            for city in range(startingCityIndex, endingCityIndex):
                cityNames.append(parentOne[city].getName())
                aChild.append(parentOne[city])

            for city in parentTwo:
                if city.getName() not in cityNames:
                    cityNames.append(city)
                    aChild.append(city)

            children.append(aChild)

        return children

    def mutateRoutes(self, routes):
        mutatedRoutes = []

        for route in routes:
            randomIndexA = random.randint(0, len(route)-1)
            randomIndexB = 0

            loopBreak = 10000
            while True:
                temp = random.randint(0, len(route)-1)

                if temp is not randomIndexA:
                    randomIndexB = temp
                    break

                loopBreak = loopBreak - 1
                if loopBreak is 0:
                    break

            tempA = route[randomIndexA]
            tempB = route[randomIndexB]

            route[randomIndexA] = tempB
            route[randomIndexB] = tempA

            mutatedRoutes.append(route)

        return mutatedRoutes

    def geneticMethod(self):
        print("Genetic Method")
        pop = self.createRandomListsOfOrderedCities(10)

        bestRoute = None

        for generations in range(0, 10000):
            rankedPop = self.rankRoutesByFitness(pop)

            currBestRoute = rankedPop[0]
            if bestRoute is None:
                bestRoute = currBestRoute
            elif self.routeFitness(currBestRoute) > self.routeFitness(bestRoute):
                bestRoute = currBestRoute

            selection = self.selection(rankedPop)
            cross = self.crossover(selection)
            mutatedRoutes = self.mutateRoutes(cross)
            pop = mutatedRoutes

        path = ""

        for city in bestRoute:
            path += city.getName() + ", "

        path += bestRoute[0].getName()

        print("Genetic Algorithm Finished")
        print("Total Distance: ", self.routeDistance(bestRoute))
        print("Path: ", path)


def main():
    graph = Graph()

    seattle = Node("seattle", [-4.5, 2])
    los_angeles = Node("los angeles", [-4, 0])
    austin = Node("austin", [-2, -2])
    denver = Node("denver", [-1.5, 0.5])
    miami = Node("miami", [2, -4])
    new_york = Node("new york", [4, 1.5])
    boston = Node("boston", [4.5, 1])

    graph.addNode(seattle)
    graph.addNode(los_angeles)
    graph.addNode(austin)
    graph.addNode(denver)
    graph.addNode(miami)
    graph.addNode(new_york)
    graph.addNode(boston)

    problem = TravelingSalesmanProblem(graph)
    problem.solve(1)

main()
